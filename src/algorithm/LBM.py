import math

from Infra.data_structure.grid import *
import taichi as ti
import numpy as np


Q = 19
dim = 3

# 邻居信息
STANDARD = 1
NO_FLUID_NEIGHBORS = 2
NO_EMPTY_NEIGHBORS = 3

FILLED = 0
EMPTIED = 1

c_s = 1.0 / math.sqrt(3)
cs_pow2 = c_s ** 2
cs2_pow4 = 2.0 * cs_pow2 * cs_pow2
cs2_pow2 = 2.0 * cs_pow2

weight = ti.field(dtype=float, shape=Q)
w = np.array([(1. / 36), (1. / 36), (2. / 36), (1. / 36), (1. / 36),
              (1. / 36), (2. / 36), (1. / 36), (2. / 36), (12. / 36),
              (2. / 36), (1. / 36), (2. / 36), (1. / 36), (1. / 36),
              (1. / 36), (2. / 36), (1. / 36), (1. / 36)], dtype=np.float32)
e = ti.field(dtype=int, shape=(Q, dim))
e_ = np.array([[0, -1, -1],
              [-1, 0, -1],
              [0, 0, -1],
              [1, 0, -1],
              [0, 1, -1],
              [-1, -1, 0],
              [0, -1, 0],
              [1, -1, 0],
              [-1, 0, 0],
              [0, 0, 0],
              [1, 0, 0],
              [-1, 1, 0],
              [0, 1, 0],
              [1, 1, 0],
              [0, -1, 1],
              [-1, 0, 1],
              [0, 0, 1],
              [1, 0, 1],
              [0, 1, 1]], dtype=np.int32)
vec19 = ti.types.vector(19, float)


@ti.data_oriented
class LBMSolver:
    def __init__(self, tau, res, gravity, smag_constant):
        weight.from_numpy(w)
        e.from_numpy(e_)

        self.nx = res[0]
        self.ny = res[1]
        self.nz = res[2]

        # field
        self.collision_field = Grid4D(res[0], res[1], res[2], Q)
        self.stream_field = Grid4D(res[0], res[1], res[2], Q)
        self.mass = RealGrid(res[0], res[1], res[2])
        self.fraction = RealGrid(res[0], res[1], res[2])
        self.flag = FlagGrid(res[0], res[1], res[2])
        self.neighbor = IntGrid(res[0], res[1], res[2])
        self.mass_change = RealGrid(res[0], res[1], res[2])  # used for distribute mass
        self.excess_mass = RealGrid(res[0], res[1], res[2])  # used for distribute mass

        # gravity
        self.gravity = ti.field(dtype=float, shape=3)
        self.gravity[0] = gravity[0]
        self.gravity[1] = gravity[1]
        self.gravity[2] = gravity[2]

        # parameter
        self.step_size = ti.field(dtype=float, shape=())
        self.step_size[None] = 1.0

        self.allow_increase = ti.field(dtype=ti.i32, shape=())
        self.allow_increase[None] = True

        self.smag_constant = smag_constant

        self.tau = ti.field(dtype=float, shape=())
        self.tau[None] = tau

        self.viscosity = ti.field(dtype=float, shape=())
        self.viscosity[None] = (self.tau[None] - 0.5) / 3.0

        self.h_collide_field = np.zeros(self.nx * self.ny * self.nz * Q)
        self.h_stream_field = np.array(self.nx * self.ny * self.nz * Q)
        self.h_mass = np.array(self.nx * self.ny * self.nz)
        self.h_fraction = np.array(self.nx * self.ny * self.nz)
        self.h_flag = np.array(self.nx * self.ny * self.nz)

    @ti.func
    def get_feq(self, density, vel):
        """
        根据密度和速度，计算feq
        """
        # feq = ti.Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        feq = vec19(0.0)
        u_dot_u = 0.0

        for i in ti.static(range(dim)):
            u_dot_u += vel[i] ** 2

        for i in ti.static(range(Q)):
            ci_dot_u = 0.0
            for j in ti.static(range(dim)):
                ci_dot_u += e[i, j] * vel[j]

            ci_dot_u_pow2 = ci_dot_u ** 2
            scale = w[i] * density
            eq = 1.0 + ci_dot_u / cs_pow2 + ci_dot_u_pow2 / cs2_pow4- u_dot_u / cs2_pow2
            feq[i] = scale * eq

        return feq

    @ti.func
    def get_density(self, i: ti.i32, j: ti.i32, k: ti.i32) -> float:
        """
        计算i，j，k网格的密度
        """
        result = 0.0
        for q in (ti.static(range(Q))):
            result += self.collision_field.data[i, j, k, q]
        return result

    @ti.func
    def get_vel(self, i: ti.i32, j: ti.i32, k: ti.i32, density: float):
        """
        计算宏观速度
        """
        vel = ti.Vector([0.0, 0.0, 0.0])
        for q in ti.static(range(Q)):
            for d in ti.static(range(dim)):
                vel[d] += self.collision_field.data[i, j, k, q] * e[q, d]

        for d in ti.static(range(dim)):
            vel[d] /= density

        return vel

    @ti.func
    def get_fraction(self, i: ti.i32, j: ti.i32, k: ti.i32) -> float:
        result = 0.0
        if self.flag.data[i, j, k] & TypeEmpty or self.flag.data[i, j, k] & TypeInterfaceToEmpty:
            result = 0.0
        elif self.flag.data[i, j, k] & TypeInterface:
            result = self.fraction.data[i, j, k]
        else:
            result = 1.0
        return result

    @ti.func
    def compute_stress_tensor(self, i: ti.i32, j: ti.i32, k: ti.i32, feq) -> float:
        magnitude = 0.0
        for alpha in ti.static(range(dim)):
            for beta in ti.static(range(dim)):
                elem = 0.0
                for q in ti.static(range(Q)):
                    elem += e[q, alpha] * e[q, beta] * (self.collision_field.data[i, j, k, q] - feq[q])
                magnitude += elem ** 2

        result = ti.math.sqrt(magnitude)
        return result

    @ti.func
    def compute_local_relaxation_time(self, tau: float, stress_tensor_norm: float) -> float:
        viscosity = (tau - 0.5) / 3.0
        smag_sqr = self.smag_constant ** 2
        stress = (ti.math.sqrt(viscosity ** 2 + 18 * smag_sqr * stress_tensor_norm) - viscosity) / \
                 (6.0 * smag_sqr)
        return 3.0 * (viscosity + smag_sqr * stress) + 0.5

    @ti.func
    def compute_post_collision_distributions(self, i: ti.i32, j: ti.i32, k: ti.i32, local_tau: float, feq):
        """
        碰撞
        """
        tau_inv = 1.0 / local_tau
        for q in ti.static(range(Q)):
            self.collision_field.data[i, j, k, q] = self.collision_field.data[i, j, k, q] - \
                                                    tau_inv * (self.collision_field.data[i, j, k, q] - feq[q])

    @ti.func
    def interpolate_empty_cell(self, i: ti.i32, j: ti.i32, k: ti.i32):
        """
        set distributions for empty cell
        """
        num_neighs = 0
        avg_density = 0.0
        avg_vel = ti.Vector([0.0, 0.0, 0.0])
        for q in ti.static(range(Q)):
            nei_i = i + e[q, 0]
            nei_j = j + e[q, 1]
            nei_k = k + e[q, 2]
            if nei_i == i and nei_j == j and nei_k == k:
                continue

            # compute average density and velocity
            if self.flag.data[nei_i, nei_j, nei_k] & (TypeFluid | TypeInterface | TypeInterfaceToFluid):
                neigh_density = self.get_density(nei_i, nei_j, nei_k)
                neigh_vel = self.get_vel(nei_i, nei_j, nei_k, neigh_density)
                num_neighs += 1
                avg_density += neigh_density
                avg_vel[0] += neigh_vel[0]
                avg_vel[1] += neigh_vel[1]
                avg_vel[2] += neigh_vel[2]

        # ti.static_assert(num_neighs != 0)
        avg_density /= num_neighs
        avg_vel[0] /= num_neighs
        avg_vel[1] /= num_neighs
        avg_vel[2] /= num_neighs

        # set fraction and distributions
        self.fraction.data[i, j, k] = self.mass.data[i, j, k] / avg_density
        feq = self.get_feq(avg_density, avg_vel)
        for q in ti.static(range(Q)):
            self.collision_field.data[i, j, k, q] = feq[q]

    @ti.func
    def compute_surface_normal(self, i: ti.i32, j: ti.i32, k: ti.i32):
        normal = ti.Vector([0.0, 0.0, 0.0])
        x_plus = self.get_fraction(i + 1, j, k)
        x_minus = self.get_fraction(i - 1, j, k)
        y_plus = self.get_fraction(i, j + 1, k)
        y_minus = self.get_fraction(i, j - 1, k)
        z_plus = self.get_fraction(i, j, k + 1)
        z_minus = self.get_fraction(i, j, k - 1)
        normal[0] = 0.5 * (x_minus - x_plus)
        normal[1] = 0.5 * (y_minus - y_plus)
        normal[2] = 0.5 * (z_minus - z_plus)
        return normal

    @ti.func
    def distribute_single_mass1(self, i: ti.i32, j: ti.i32, k: ti.i32, change_type: ti.i32):
        cur_excess_mass = 0.0
        density = self.get_density(i, j, k)
        if change_type == FILLED:
            cur_excess_mass = self.mass.data[i, j, k] - density
            self.mass_change.data[i, j, k] -= cur_excess_mass
        else:
            cur_excess_mass = self.mass.data[i, j, k]
            self.mass_change.data[i, j, k] -= cur_excess_mass
        self.excess_mass.data[i, j, k] = cur_excess_mass

    @ti.func
    def distribute_single_mass2(self, i: ti.i32, j: ti.i32, k: ti.i32, change_type: ti.i32, excess_mass: float):
        normal = self.compute_surface_normal(i, j, k)
        weights = vec19(0.0)
        weights_backup = vec19(0.0)

        # Calculate the unnormalized weights.
        for q in ti.static(range(Q)):
            nei_i = i + e[q, 0]
            nei_j = j + e[q, 1]
            nei_k = k + e[q, 2]
            if i == nei_i and j == nei_j and k == nei_k or ~(self.flag.data[nei_i, nei_j, nei_k] & TypeInterface):
                continue

            weights_backup[q] = 1.0
            dot_product = normal[0] * e[q, 0] + normal[1] * e[q, 1] + normal[2] * e[q, 2]

            if change_type == FILLED:
                weights[q] = ti.math.max(0.0, dot_product)
            else:
                weights[q] = -ti.math.min(0.0, dot_product)
            # ti.static_assert(weights[q] >= 0.0)

        # calculate normalizer
        normalizer = 0.0
        for q in ti.static(range(Q)):
            normalizer += weights[q]

        if normalizer == 0.0:
            weights = weights_backup
            for q in ti.static(range(Q)):
                normalizer += weights[q]
        if normalizer != 0.0:
            # redistribute weights
            for q in ti.static(range(Q)):
                nei_i = i + e[q, 0]
                nei_j = j + e[q, 1]
                nei_k = k + e[q, 2]
                self.mass_change[nei_i, nei_j, nei_k] += weights[q] / normalizer * excess_mass

    @ti.func
    def inverse_q(self, q: ti.i32) -> ti.i32:
        ti.static_assert(0 <= q < Q)
        return (Q - 1) - q

    @ti.func
    def calculate_se(self, i: ti.i32, j: ti.i32, k: ti.i32, q: ti.i32) -> float:
        result = 0.0
        inv = self.inverse_q(q)
        nei_i = i + e[q, 0]
        nei_j = j + e[q, 1]
        nei_k = k + e[q, 2]
        if self.neighbor.data[i, j, k] == self.neighbor.data[nei_i, nei_j, nei_k]:
            result = self.collision_field.data[nei_i, nei_j, nei_k, inv] - self.collision_field.data[i, j, k, q]
        elif (self.neighbor.data[nei_i, nei_j, nei_k] == STANDARD and self.neighbor.data[i, j, k] == NO_FLUID_NEIGHBORS) \
                or (self.neighbor.data[nei_i, nei_j, nei_k] == NO_EMPTY_NEIGHBORS and
                (self.neighbor.data[i, j, k] == STANDARD or self.neighbor.data[i, j, k] == NO_FLUID_NEIGHBORS)):
            result = -self.collision_field.data[i, j, k, q]
        else:
            result = self.collision_field.data[nei_i, nei_j, nei_k, inv]
        return result

    @ti.func
    def single_no_slip(self, i: ti.i32, j: ti.i32, k: ti.i32):
        # ti.static_assert(self.flag.data[i, j, k] & TypeNoSlip)
        for q in ti.static(range(Q)):
            nei_i = i + e[q, 0]
            nei_j = j + e[q, 1]
            nei_k = k + e[q, 2]
            if nei_i <= 0 or nei_j <= 0 or nei_k <= 0 or nei_i >= self.nx - 1 or nei_j >= self.ny - 1 or nei_k >= self.nz - 1:
                continue
            inv = self.inverse_q(q)
            self.collision_field.data[i, j, k, q] = self.collision_field.data[nei_i, nei_j, nei_k, inv]
            if self.flag.data[nei_i, nei_j, nei_k] & TypeEmpty:
                self.collision_field.data[i, j, k, q] = 0.0

    @ti.kernel
    def collide(self):
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            if self.flag.data[i, j, k] & (TypeFluid | TypeInterface):
                # compute current density and velocity
                cur_density = self.get_density(i, j, k)
                cur_vel = self.get_vel(i, j, k, cur_density)

                # add gravity
                for d in ti.static(range(dim)):
                    cur_vel[d] += self.gravity[d] * self.tau[None]

                # compute feq by current vel
                feq = self.get_feq(cur_density, cur_vel)

                # user turbulence model
                local_tau = 0.0
                if self.smag_constant > 0.0:
                    stress_norm = self.compute_stress_tensor(i, j, k, feq)
                    local_tau = self.compute_local_relaxation_time(self.tau[None], stress_norm)
                else:
                    local_tau = self.tau[None]

                # compute post collision distributions
                self.compute_post_collision_distributions(i, j, k, local_tau, feq)

                # set mass and fraction
                if self.flag.data[i, j, k] & TypeFluid:
                    self.mass.data[i, j, k] = cur_density
                else:
                    self.fraction.data[i, j, k] + self.mass.data[i, j, k] / cur_density

    @ti.kernel
    def get_potential_updates(self):
        offset = 0.001
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            # deal with interface cell only
            if self.flag.data[i, j, k] & TypeInterface:
                cur_density = 0.0
                if self.fraction.data[i, j, k] >= 0.0:
                    cur_density = self.mass.data[i, j, k] / self.fraction.data[i, j, k]

                # set intermediate cell type
                if self.mass.data[i, j, k] > (1.0 + offset) * cur_density:
                    self.flag.data[i, j, k] &= ~TypeInterface
                    self.flag.data[i, j, k] |= TypeInterfaceToFluid
                elif self.mass.data[i, j, k] < -offset * cur_density:
                    self.flag.data[i, j, k] &= ~TypeInterface
                    self.flag.data[i, j, k] |= TypeInterfaceToEmpty

                # deal with artifacts
                if self.neighbor.data[i, j, k] == NO_FLUID_NEIGHBORS and self.mass.data[i, j, k] < 0.1 * cur_density:
                    self.flag.data[i, j, k] &= ~TypeInterface
                    self.flag.data[i, j, k] |= TypeInterfaceToEmpty
                if self.neighbor.data[i, j, k] == NO_EMPTY_NEIGHBORS and self.mass.data[i, j, k] > 0.9 * cur_density:
                    self.flag.data[i, j, k] &= ~TypeInterface
                    self.flag.data[i, j, k] |= TypeInterfaceToFluid

    @ti.kernel
    def flag_reinit(self):
        # deal with interface to fluid firstly
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            if self.flag.data[i, j, k] & TypeInterfaceToFluid:
                for q in ti.static(range(Q)):
                    nei_i = i + e[q, 0]
                    nei_j = j + e[q, 1]
                    nei_k = k + e[q, 2]
                    if i == nei_i and j == nei_j and k == nei_k:
                        continue

                    if self.flag.data[nei_i, nei_j, nei_k] & TypeEmpty:
                        self.flag.data[nei_i, nei_j, nei_k] &= ~TypeEmpty
                        self.flag.data[nei_i, nei_j, nei_k] |= TypeInterface
                        self.mass.data[nei_i, nei_j, nei_k] = 0.0
                        self.fraction.data[nei_i, nei_j, nei_k] = 0.0
                        self.interpolate_empty_cell(nei_i, nei_j, nei_k)
                    elif self.flag.data[nei_i, nei_j, nei_k] & TypeInterfaceToEmpty:
                        self.flag.data[nei_i, nei_j, nei_k] &= ~TypeInterfaceToEmpty
                        self.flag.data[nei_i, nei_j, nei_k] |= TypeInterface

        # deal with interface to empty secondly
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            if self.flag.data[i, j, k] & TypeInterfaceToEmpty:
                for q in ti.static(range(Q)):
                    nei_i = i + e[q, 0]
                    nei_j = j + e[q, 1]
                    nei_k = k + e[q, 2]
                    if i == nei_i and j == nei_j and k == nei_k:
                        continue

                    if self.flag.data[nei_i, nei_j, nei_k] & TypeFluid:
                        self.flag.data[nei_i, nei_j, nei_k] &= ~TypeFluid
                        self.flag.data[nei_i, nei_j, nei_k] |= TypeInterface
                        self.mass.data[nei_i, nei_j, nei_k] = self.get_density(nei_i, nei_j, nei_k)
                        self.fraction.data[nei_i, nei_j, nei_k] = 1.0

    @ti.kernel
    def distribute_mass(self):
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            if self.flag.data[i, j, k] & TypeInterfaceToFluid:
                self.distribute_single_mass1(i, j, k, FILLED)
            elif self.flag.data[i, j, k] & TypeInterfaceToEmpty:
                self.distribute_single_mass1(i, j, k, EMPTIED)

        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            if self.flag.data[i, j, k] & TypeInterfaceToFluid:
                self.distribute_single_mass2(i, j, k, FILLED, self.excess_mass[i, j, k])
            elif self.flag.data[i, j, k] & TypeInterfaceToEmpty:
                self.distribute_single_mass2(i, j, k, EMPTIED, self.excess_mass[i, j, k])

        # clear temporary cell type
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            if self.flag.data[i, j, k] & TypeInterfaceToFluid:
                self.flag.data[i, j, k] &= ~TypeInterfaceToFluid
                self.flag.data[i, j, k] |= TypeFluid
            elif self.flag.data[i, j, k] & TypeInterfaceToEmpty:
                self.flag.data[i, j, k] &= ~TypeInterfaceToEmpty
                self.flag.data[i, j, k] |= TypeEmpty

        # distribute mass
        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            cur_density = self.get_density(i, j, k)
            self.mass.data[i, j, k] += self.mass_change.data[i, j, k]
            self.fraction.data[i, j, k] = self.mass.data[i, j, k] / cur_density

    @ti.kernel
    def adapt_timestep(self):
        # calculate the max velocity
        max_vel_norm = 0.0
        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            if self.flag.data[i, j, k] & (TypeFluid | TypeInterface):
                cur_density = self.get_density(i, j, k)
                cur_vel = self.get_vel(i, j, k, cur_density)
                cur_norm = ti.math.sqrt(cur_vel[0] ** 2 + cur_vel[1] ** 2 + cur_vel[2] ** 2)
                ti.atomic_max(max_vel_norm, cur_norm)

        critical_vel = 0.5 * c_s * c_s
        multiplier = 4.0 / 5.0
        upper_limit = critical_vel / multiplier
        lower_limit = critical_vel * multiplier

        is_adapt = True
        old_timestep = self.step_size[None]
        old_tau = self.tau[None]
        new_tau = 0.0
        new_timestep = old_timestep
        if max_vel_norm > upper_limit:
            new_timestep *= multiplier
        elif max_vel_norm < lower_limit and self.allow_increase[None]:
            new_timestep /= multiplier
        else:
            is_adapt = False
            new_tau = old_tau
            new_timestep = old_timestep

        time_ratio = 1.0
        if is_adapt:
            time_ratio = new_timestep / old_timestep
            new_tau = time_ratio * (old_tau - 0.5) + 0.5
            min_tau = 0.0
            if self.smag_constant > 0.0:
                min_tau = 0.5
            else:
                min_tau = 1.0 / 1.99

            if new_tau <= min_tau:
                is_adapt = False
                new_tau = old_tau
                new_timestep = old_timestep

        if is_adapt:
            # rescale gravity
            self.gravity[0] = self.gravity[0] * time_ratio * time_ratio
            self.gravity[1] = self.gravity[1] * time_ratio * time_ratio
            self.gravity[2] = self.gravity[2] * time_ratio * time_ratio

            # rescale mass
            total_fluid_volume = 0.0
            total_mass = 0.0
            # calculate total mass and fluid volume
            for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
                if self.flag.data[i, j, k] & TypeFluid:
                    total_mass += self.mass.data[i, j, k]   # atomic
                    total_fluid_volume += 1.0
                elif self.flag.data[i, j, k] & TypeInterface:
                    total_mass += self.mass.data[i, j, k]   # atomic
                    total_fluid_volume += self.fraction.data[i, j, k]

            median_density = total_fluid_volume / total_mass

            for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
                if self.flag.data[i, j, k] & (TypeFluid | TypeInterface):
                    old_density = self.get_density(i, j, k)
                    new_density = time_ratio * (old_density - median_density) + median_density

                    old_vel = self.get_vel(i, j, k, old_density)
                    new_vel = old_vel
                    new_vel *= time_ratio

                    old_feq = self.get_feq(old_density, old_vel)
                    new_feq = self.get_feq(new_density, new_vel)

                    # use turbulent model
                    tau_ratio = 0.0
                    if self.smag_constant > 0.0:
                        old_stress = self.compute_stress_tensor(i, j, k, old_feq)
                        old_local_tau = self.compute_local_relaxation_time(old_tau, old_stress)
                        new_stress = self.compute_stress_tensor(i, j, k, old_feq)   # old_feq
                        new_local_tau = self.compute_local_relaxation_time(new_tau, new_stress)
                        tau_ratio = time_ratio * (new_local_tau / old_local_tau)
                    else:
                        tau_ratio = time_ratio * (new_tau / old_tau)

                    for q in ti.static(range(Q)):
                        feq_ratio = new_feq[q] / old_feq[q]
                        self.collision_field.data[i, j, k, q] = \
                            feq_ratio * (old_feq[q] + tau_ratio * (self.collision_field.data[i, j, k, q] - old_feq[q]))

                    if self.flag.data[i, j, k] & TypeInterface:
                        self.mass.data[i, j, k] *= old_density / new_density
                        self.fraction.data[i, j, k] = self.mass.data[i, j, k] / new_density

        self.tau[None] = new_tau
        self.step_size[None] = new_timestep

    @ti.kernel
    def treat_boundary(self):
        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            flag = self.flag.data[i, j, k]
            if flag & TypeFreeSlip:
                pass
            elif flag & TypeNoSlip:
                self.single_no_slip(i, j, k)

    @ti.kernel
    def stream(self):
        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            # steam fluid and interface cell
            if self.flag.data[i, j, k] & (TypeFluid | TypeInterface):
                for q in ti.static(range(Q)):
                    nei_i = i - e[q, 0]
                    nei_j = j - e[q, 1]
                    nei_k = k - e[q, 2]
                    self.stream_field.data[i, j, k, q] = self.collision_field.data[nei_i, nei_j, nei_k, q]
                    # ti.static_assert(self.stream_field.data[i, j, k, q] >= 0.0)

        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            # set distributions for interface cell
            if self.flag.data[i, j, k] & TypeInterface:
                normal = self.compute_surface_normal(i, j, k)
                has_fluid_neighbors = False
                has_empty_neighbors = False
                for q in ti.static(range(Q)):
                    nei_i = i - e[q, 0]
                    nei_j = j - e[q, 1]
                    nei_k = k - e[q, 2]
                    if nei_i == i and nei_j == j and nei_k == k:
                        continue

                    is_empty_adjacent = self.flag.data[nei_i, nei_j, nei_k] & TypeEmpty
                    has_fluid_neighbors = has_fluid_neighbors or self.flag.data[nei_i, nei_j, nei_k] & TypeFluid
                    has_empty_neighbors = has_empty_neighbors or self.flag.data[nei_i, nei_j, nei_k] & TypeEmpty

                    inv = self.inverse_q(q)
                    dot_product = normal[0] * e[inv, 0] + normal[1] * e[inv, 1] + normal[2] * e[inv, 2]
                    is_normal_direction = dot_product > 0.0
                    # reconstruct distributions for normal directions and empty-cell directions
                    if is_empty_adjacent or is_normal_direction:
                        atmospheric_pressure = 1.0
                        cur_density = self.get_density(i, j, k)
                        cur_vel = self.get_vel(i, j, k, cur_density)
                        feq = self.get_feq(atmospheric_pressure, cur_vel)
                        self.stream_field.data[i, j, k, q] = feq[inv] + feq[q] - self.collision_field.data[i, j, k, inv]

                # set neighbors information
                is_standard_cell = has_empty_neighbors and has_fluid_neighbors
                if is_standard_cell:
                    self.neighbor.data[i, j, k] = STANDARD
                elif not has_fluid_neighbors:
                    self.neighbor.data[i, j, k] = NO_FLUID_NEIGHBORS
                elif not has_empty_neighbors:
                    self.neighbor.data[i, j, k] = NO_EMPTY_NEIGHBORS

    @ti.kernel
    def stream_mass(self):
        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            delta_mass = 0.0
            # stream mass only for interface cell
            if self.flag.data[i, j, k] & TypeInterface:
                cur_fraction = self.get_fraction(i, j, k)
                for q in ti.static(range(Q)):
                    nei_i = i + e[q, 0]
                    nei_j = j + e[q, 1]
                    nei_k = k + e[q, 2]
                    if nei_i == i and nei_j == j and nei_k == k:
                        continue
                    if self.flag.data[nei_i, nei_j, nei_k] & TypeFluid:
                        inv = self.inverse_q(q)
                        delta_mass += self.collision_field.data[nei_i, nei_j, nei_k, inv] - \
                                      self.collision_field.data[i, j, k, q]
                    elif self.flag.data[nei_i, nei_j, nei_k] & TypeInterface:
                        nei_fraction = self.get_fraction(nei_i, nei_j, nei_k)
                        s_e = self.calculate_se(i, j, k, q)
                        delta_mass += 0.5 * s_e * (cur_fraction + nei_fraction)

            self.mass.data[i, j, k] += delta_mass

    @ti.kernel
    def swap_field(self):
        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            for q in ti.static(range(Q)):
                temp = self.collision_field.data[i, j, k, q]
                self.collision_field.data[i, j, k, q] = self.stream_field.data[i, j, k, q]
                self.stream_field.data[i, j, k, q] = temp

    @ti.kernel
    def init_scene(self):
        # init collide and stream field
        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            for q in ti.static(range(Q)):
                self.collision_field.data[i, j, k, q] = weight[q]
                self.stream_field.data[i, j, k, q] = weight[q]

        # init fluid
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            if 1 <= i <= 30 and 1 <= j <= 30:
                self.flag.data[i, j, k] = TypeFluid
            else:
                self.flag.data[i, j, k] = TypeEmpty

        # init boundary cell flag
        boundary_type = TypeNoSlip
        for i in range(self.nx):
            for j in range(self.ny):
                self.flag.data[i, j, 0] = boundary_type
                self.flag.data[i, j, self.nz - 1] = boundary_type

        for j in range(self.ny):
            for k in range(self.nz):
                self.flag.data[0, j, k] = boundary_type
                self.flag.data[self.nx - 1, j, k] = boundary_type

        for k in range(self.nz):
            for i in range(self.nx):
                self.flag.data[i, 0, k] = boundary_type
                self.flag.data[i, self.ny - 1, k] = boundary_type

        # init mass and fraction
        for i, j, k in ti.ndrange((0, self.nx), (0, self.ny), (0, self.nz)):
            if self.flag.data[i, j, k] & TypeFluid:
                self.mass.data[i, j, k] = 1.0
                self.fraction.data[i, j, k] = 1.0
            elif self.flag.data[i, j, k] & TypeInterface:
                self.mass.data[i, j, k] = 0.5
                self.fraction.data[i, j, k] = 0.5
            else:
                self.mass.data[i, j, k] = 0.0
                self.fraction.data[i, j, k] = 0.0

        # full interface
        for i, j, k in ti.ndrange((1, self.nx - 1), (1, self.ny - 1), (1, self.nz - 1)):
            if self.flag.data[i, j, k] & TypeFluid:
                for q in ti.static(range(Q)):
                    nei_i = i + e[q, 0]
                    nei_j = j + e[q, 1]
                    nei_k = k + e[q, 2]
                    if self.flag.data[nei_i, nei_j, nei_k] & TypeEmpty:
                        self.flag.data[nei_i, nei_j, nei_k] &= ~TypeEmpty
                        self.flag.data[nei_i, nei_j, nei_k] |= TypeInterface
                        self.mass.data[nei_i, nei_j, nei_k] = 0.5
                        self.fraction.data[nei_i, nei_j, nei_k] = 0.5

    def copy_data_to_host(self):
        self.h_collide_field = self.collision_field.data.to_numpy()
        self.h_stream_field = self.stream_field.data.to_numpy()
        self.h_mass = self.mass.data.to_numpy()
        self.h_fraction = self.fraction.data.to_numpy()
        self.h_flag = self.flag.data.to_numpy()

    def simulate(self, timestep):
        t = 1.0
        real_time_steps = 0
        increase_next = 1
        increase_delay = int(math.pow(self.nx * self.ny * self.nz, 1.0 / 3.0) * 4.0)
        while t < timestep:
            print(t)
            real_time_steps += 1
            self.collide()
            # self.copy_data_to_host()

            self.get_potential_updates()
            # self.copy_data_to_host()

            self.flag_reinit()
            # self.copy_data_to_host()

            self.mass_change.set_const(0.0)
            self.excess_mass.set_const(0.0)
            self.distribute_mass()
            # self.copy_data_to_host()

            step_size_before = self.step_size[None]
            self.allow_increase[None] = real_time_steps > increase_next
            self.adapt_timestep()
            if self.step_size[None] < step_size_before:
                increase_next = real_time_steps + increase_delay

            # self.copy_data_to_host()

            self.treat_boundary()
            # self.copy_data_to_host()

            self.stream()
            # self.copy_data_to_host()

            self.stream_mass()
            # self.copy_data_to_host()

            self.swap_field()
            # self.copy_data_to_host()

            t += self.step_size[None]



