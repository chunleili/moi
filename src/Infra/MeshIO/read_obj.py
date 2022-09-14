
# API from meshio
def read_obj(filename):
    '''
    调用meshio的API来读取obj网格
    
    输入：
        filename: OBJ文件名
    输出：
        points: 顶点位置
        cells: 连接关系
    用法示例：见test_obj.py
    '''
    import meshio.obj as obj
    mesh = obj.read(filename)
    points = mesh.points
    cells = mesh.cells
    return points, cells