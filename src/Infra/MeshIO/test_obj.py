from pathlib import Path

'''先读入OBJ'''
from ..read_obj import read_obj
points, cells = read_obj("MeshIO\Dragon_50k.obj")

'''再输出OBJ'''
from ..write_obj import write_obj
write_obj(points,cells,"test_output.obj")
