import taichi as ti


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
class GridReal(GridBase):
    """
    实数场
    """
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32):
        super().__init__(size_x, size_y, size_z, float)
        ti.root.dense(ti.ijk, (size_x, size_y, size_z)).place(self.data)


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
class Grid4D(GridBase):
    """
    4维场
    """
    def __init__(self, size_x: ti.i32, size_y: ti.i32, size_z: ti.i32, q: ti.i32):
        super().__init__(size_x, size_y, size_z, float)
        self.q = q
        ti.root.dense(ti.ijkl, (size_x, size_y, size_z, q)).place(self.data)


