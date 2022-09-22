def find_boundingBox(pos):
    '''
    输入一个点云数组，输出boundingBox(即最小点和最大点坐标)
    '''
    min_x = min(pos[:][0])
    min_y = min(pos[:][1])
    min_z = min(pos[:][2])

    max_x = max(pos[:][0])
    max_y = max(pos[:][1])
    max_z = max(pos[:][2])

    return [min_x, min_y, min_z, max_x, max_y, max_z]