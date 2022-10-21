from grid import GridInt
from grid import GridReal
from grid import GridVec3
from grid import GridMac
import taichi as ti

ti.init(arch=ti.cuda)

def test_grid():
    grid_int1 = GridInt(3, 4, 5)
    grid_int1.set_const(3)
    grid_int2 = GridInt(3, 4, 5)
    grid_int2.set_const(5)
    grid_int2.join(grid_int1)

    grid_real1 = GridReal(2, 2, 9)
    grid_real1.set_const(3.3)
    grid_real2 = GridReal(2, 2, 9)
    grid_real2.set_const(4.4)
    grid_real2.add_const(1.1)
    grid_real2.join(grid_real1)

    grid_vec3_1 = GridVec3(2, 3, 4)
    grid_vec3_1.set_const(ti.Vector([2.0, 3.0, 4.0]))
    grid_vec3_1.print_data()
    grid_vec3_2 = GridVec3(2, 3, 4)
    grid_vec3_2.set_const(ti.Vector([5.0, 4.0, 4.0]))
    grid_vec3_2.print_data()
    grid_vec3_2.join(grid_vec3_1)
    grid_vec3_2.print_data()

    grid_mac = GridMac(4, 5, 6)
    grid_mac.x[2, 2, 2] = 3

    grid_real1.data[1, 1, 1] = 2



test_grid()
