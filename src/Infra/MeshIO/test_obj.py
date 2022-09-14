"""
直接运行这个文件来测试OBJ的读取和输出！

请先cd到这个文件所在的目录，然后用
python test_obj.py

你将会看到输出一个名为test_output.obj的文件，
它就是我们读入Dragon_50k.obj之后又写出来的！
"""


'''先读入OBJ'''
from read_obj import read_obj
points, cells = read_obj("Dragon_50k.obj")

'''再输出OBJ'''
from write_obj import write_obj
write_obj(points,cells,"test_output.obj")
