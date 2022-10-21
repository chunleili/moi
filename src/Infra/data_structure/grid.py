import taichi as ti


TypeNone = 2 << 0
TypeFluid = 2 << 1
TypeObstacle = 2 << 2
TypeEmpty = 2 << 3
TypeInflow = 2 << 4
TypeOutflow = 2 << 5
TypeOpen = 2 << 6
TypeInterface = 2 << 7
TypeFreeSlip = 2 << 8   # for LBM
TypeNoSlip = 2 << 9     # for LBM
TypeInterfaceToFluid = 2 << 10
TypeInterfaceToEmpty = 2 << 11


@ti.data_oriented
class GridBase:
    """
    网格基类

    Attributes:
        size_x:
        size_y:
        size_z:
        grid_size: 网格中元素总数
        data: 数据容器
    """
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32, data_type: ti.template()):
        """
        Args:
            size_x: x维度大小
            size_y: y维度大小
            size_z: z维度大小
            data_type: 网格中存放的数据类型。如int, float。
        """
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.grid_size = size_x * size_y * size_z
        self.data = ti.field(dtype=data_type)

    @ti.func
    def get_index(self, i: ti.i32, j: ti.i32, k: ti.i32):
        """
        根据位置，获得索引

        Args:
            i: 下标i
            j: 下标j
            k: 下标k
        """
        return self.size_x * self.size_y * k + self.size_x * j + i

    @ti.func
    def check_ijk(self, i: ti.i32, j: ti.i32, k: ti.i32):
        """
        检查下标是否合法
        """
        result = True
        if i < 0 or i > self.size_x - 1:
            result = False
        elif j < 0 or j > self.size_y - 1:
            result = False
        elif k < 0 or k > self.size_z - 1:
            result = False
        return result

    @ti.kernel
    def set_const(self, val: ti.template()):
        """
        设置每个网格元素的值为val

        Args:
            val: 要设置的值

        """
        for i in ti.grouped(self.data):
            self.data[i] = val

    @ti.kernel
    def join(self, b: ti.template()):
        """
        取2个场相同位置的最小值

        Args:
            b: 另一个与该场shape相同的场

        """
        ti.static_assert(self.data.shape == b.data.shape, 'The fields have different shape.')
        assert self.data.dtype == b.data.dtype
        for i, j, k in self.data:
            self.data[i, j, k] = min(self.data[i, j, k], b.data[i, j, k])

    @ti.kernel
    def add_const(self, val: ti.template()):
        """
        将场加一个常数

        Args:
            val: 要加的常数值

        """
        for i in ti.grouped(self.data):
            self.data[i] += val

    @ti.func
    def __getitem__(self, i, j, k):
        return self.data[i, j, k]

    @ti.kernel
    def print_data(self):
        for i in ti.grouped(self.data):
            print(self.data[i])


@ti.data_oriented
class GridInt(GridBase):
    """
    整数场
    """
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32):
        super().__init__(size_x, size_y, size_z, int)
        ti.root.dense(ti.ijk, (size_x, size_y, size_z)).place(self.data)


@ti.data_oriented
class GridFlag(GridInt):
    """
    网格类型场
    """
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32):
        super().__init__(size_x, size_y, size_z)

    @ti.func
    def is_fluid(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeFluid

    @ti.func
    def is_empty(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeEmpty

    @ti.func
    def is_obstacle(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeObstacle

    @ti.func
    def is_outflow(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeOutflow

    @ti.func
    def is_inflow(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeInflow

    @ti.func
    def is_interface(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeInterface

    @ti.func
    def is_open(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeOpen

    @ti.func
    def is_free_slip(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeFreeSlip

    @ti.func
    def is_no_slip(self, i: ti.i32, j: ti.i32, k: ti.i32) -> bool:
        return self.data[i, j, k] & TypeNoSlip


@ti.data_oriented
class GridReal(GridBase):
    """
    实数场
    """
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32):
        super().__init__(size_x, size_y, size_z, float)
        ti.root.dense(ti.ijk, (size_x, size_y, size_z)).place(self.data)


@ti.data_oriented
class Grid4D(GridBase):
    """
    4维场
    """
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32, q: ti.i32):
        super().__init__(size_x, size_y, size_z, float)
        self.q = q
        ti.root.dense(ti.ijkl, (size_x, size_y, size_z, q)).place(self.data)


@ti.data_oriented
class GridVec3(GridBase):
    """
    三维场
    """
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.data = ti.Vector.field(3, dtype=float, shape=(size_x, size_y, size_z))

    @ti.kernel
    def join(self, b: ti.template()):
        """
        两个场取向量的最小值

        Args:
            b: 一个与该场shape相同的3维向量场

        """
        ti.static_assert(self.data.shape == b.data.shape, 'The fields have different shape.')
        for i, j, k in self.data:
            length_a = ti.math.length(self.data[i, j, k])
            length_b = ti.math.length(b.data[i, j, k])
            if length_b < length_a:
                self.data[i, j, k] = b.data[i, j, k]


@ti.data_oriented
class GridMac(GridBase):
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32):
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.x = ti.field(dtype=float, shape=(size_x + 1, size_y, size_z))
        self.y = ti.field(dtype=float, shape=(size_x, size_y + 1, size_z))
        self.z = ti.field(dtype=float, shape=(size_x, size_y, size_z + 1))

    @ti.func
    def get_mac_x(self, i: ti.i32, j: ti.i32, k: ti.i32):
        """
        获得x场i，j，k位置的3维值
        """
        ti.static_assert(self.check_ijk(i, j, k))
        val = ti.Vector([0.0, 0.0, 0.0])
        val[0] = self.x[i, j, k]
        val[1] = (self.y[i, j, k] + self.y[i, j + 1, k] + self.y[i - 1, j, k] + self.y[i - 1, j + 1, k]) * 0.25
        val[2] = (self.z[i, j, k] + self.z[i, j, k + 1] + self.z[i - 1, j, k] + self.z[i - 1, j, k + 1]) * 0.25
        return val

    @ti.func
    def get_mac_y(self, i: ti.i32, j: ti.i32, k: ti.i32):
        """
        获得y场i，j，k位置的3维值
        """
        ti.static_assert(self.check_ijk(i, j, k))
        val = ti.Vector([0.0, 0.0, 0.0])
        val[0] = (self.x[i, j - 1, k] + self.x[i + 1, j - 1, k] + self.x[i, j, k] + self.x[i + 1, j, k]) * 0.25
        val[1] = self.y[i, j, k]
        val[2] = (self.z[i, j - 1, k] + self.z[i, j - 1, k + 1] + self.z[i, j, k] + self.z[i, j, k + 1]) * 0.25
        return val

    @ti.func
    def get_mac_z(self, i: ti.i32, j: ti.i32, k: ti.i32):
        """
        获得z场i，j，k位置的3维值
        """
        ti.static_assert(self.check_ijk(i, j, k))
        val = ti.Vector([0.0, 0.0, 0.0])
        val[0] = (self.x[i, j, k - 1] + self.x[i + 1, j, k - 1] + self.x[i, j, k] + self.x[i + 1, j, k]) * 0.25
        val[1] = (self.y[i, j, k - 1] + self.y[i, j + 1, k - 1] + self.y[i, j, k] + self.y[i, j + 1, k]) * 0.25
        val[2] = self.z[i, j, k]
        return val

    @ti.func
    def get_center_val(self, i: ti.i32, j: ti.i32, k: ti.i32):
        """
        获得中心网格（cell-centered grid）i，j，k位置的值
        """
        val = ti.Vector([0.0, 0.0, 0.0])
        val[0] = (self.x[i, j, k] + self.x[i + 1, j, k]) * 0.5
        val[1] = (self.y[i, j, k] + self.y[i, j + 1, k]) * 0.5
        val[2] = (self.z[i, j, k] + self.z[i, j, k + 1]) * 0.5
        return val
