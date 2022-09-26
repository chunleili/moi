import numpy as np
import mcubes as mcubes
import MeshIO.read_obj as robj

def test_surf_recon():
    #首先读入点数据到numpy数组
    points,_ = robj.read_obj("mpm_fluid.obj")


if __name__ == "__main__":
    test_surf_recon