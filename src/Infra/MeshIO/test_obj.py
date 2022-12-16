"""
直接运行这个文件来测试OBJ的读取和输出！

请先cd到这个文件所在的目录，然后用
python test_obj.py

你将会看到输出一个名为output_obj.obj的文件，
它就是我们读入Dragon_50k.obj之后又写出来的！
同时还会写出一个名为output_obj.txt的文件，也是我们读入后写出来的。
"""
import numpy as np

# 先读入OBJ
from read_obj import read_obj
points, cells = read_obj("Dragon_50k.obj")

np.savetxt("output_obj.txt",points)
print("读入obj完毕，输出output_obj.txt")


# 再输出OBJ
from write_obj import write_obj
write_obj(points,cells,"output_obj.obj")
print("读入并输出obj完毕，输出output_obj.obj")
