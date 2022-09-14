def write_obj(points, cells, filename):
    '''
    导出到obj格式文件。
    
    输入参数：
        points: 顶点位置（3维）
        cells: 表面三角形顶点连接关系（哪三个点构成一个三角形？顶点编号按照顺序排列）
        filename: 导出的文件名
    用法示例：见test_obj.py
    '''
    with open(filename, 'w') as f:
        for p in points:
            f.write("v {} {} {}\n".format(*p))
        for pID in cells[0].data:
            f.write("f {} {} {}\n".format(*(pID + 1)))