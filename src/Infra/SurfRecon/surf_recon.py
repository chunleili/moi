from tokenize import Double
import numpy as np
import mcubes

def surf_recon(pos, dx: float):
    '''
        流体表面重建。
        
        输入： 
        pos: 3维点坐标数组。3维numpy数组。
        dx: 所重建出来的网格的尺寸。这个数值越大，所重建的网格精度越高。

        输出：surf 三角面。也是3维numpy数组。存储的是pos的编号（下标），因此都是整数。
    '''
    #先转换数据格式, 建立背景网格
    #num_x..是背景网格的大小
    num_x,num_y,num_z = _find_boundingBox(pos, dx)
    X, Y, Z = np.mgrid[:num_x, :num_y, :num_z]

    #把点云数据存入到背景网格中
    

    #然后利用mcubes的算法
    vertices, surf = mcubes.marching_cubes(u, 0)
    return surf



#遍历点云，从而为点云确定背景网格的大小。即寻找其bounding box
def  _find_boundingBox(pos,dx):
    min_x = min(pos[:][0])
    min_y = min(pos[:][1])
    min_z = min(pos[:][2])

    max_x = max(pos[:][0])
    max_y = max(pos[:][1])
    max_z = max(pos[:][2])

    num_x = int((max_x - min_x)/dx) + 1
    num_y = int((max_y - min_y)/dx) + 1
    num_z = int((max_z - min_z)/dx) + 1

    return num_x, num_y, num_z
