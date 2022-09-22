"""
直接运行这个文件来测试ply的读取和输出！

请先cd到这个文件所在的目录，然后用
python test_ply.py

你将会看到输出一个名为output_ply.ply的文件，
它就是我们读入mpm3d.ply之后又写出来的！
"""

import numpy as np

#先读入ply
from read_ply import read_ply
points, cells = read_ply("E:\Dev\moi\moi\src\Infra\MeshIO\mpm3d.ply")

#打印结果
np.savetxt("output_ply.txt", points)
print("读入ply完毕，输出output_ply.txt")

#打印BoundingBox
from find_boundingBox import find_boundingBox
min_x, min_y, min_z, max_x, max_y, max_z = find_boundingBox(points)
print(f"min_x: {min_x}\nmin_y: {min_y}\nmin_z: {min_z}\nmax_x: {max_x}\nmax_y: {max_y}\nmax_z: {max_z}\n")

# # 再输出ply（尚未完成）
# from write_ply import write_ply
# write_ply(points,cells,"output_ply.ply")
