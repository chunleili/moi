def write_ply(points, cells, filename, binary: bool=False):
    '''
    导出到ply格式文件。(Working In Progress)
    
    输入参数：
        points: 顶点位置（3维）
        cells: 表面三角形顶点连接关系（哪三个点构成一个三角形？顶点编号按照顺序排列）
        filename: 导出的文件名
    用法示例：见test_ply.py
    '''
    # FIXME: 目前尚未完成
    
    # import meshio
    # cells_ = [
    # ("triangle", cells)
    # ]

    # #首先定义meshio需要的数据结构Mesh
    # mesh = meshio.Mesh(
    #     points,
    #     cells_,
    # )
    # meshio.ply.write(filename, mesh, binary)