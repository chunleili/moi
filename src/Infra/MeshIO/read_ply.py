# API from meshio
def read_ply(filename):
    '''
    调用meshio的API来读取ply网格
    
    输入：
        filename: ply文件名
    输出：
        points: 顶点位置
        cells: 连接关系
    用法示例：见test_ply.py
    '''
    import meshio.ply as ply
    mesh = ply.read(filename)
    points = mesh.points
    cells = mesh.cells
    return points, cells