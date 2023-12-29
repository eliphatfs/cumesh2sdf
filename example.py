import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# mesh = trimesh.load("tmp/spot.obj")
mesh = trimesh.load(r"C:\Users\eliphat\Downloads\37730.stl")
tris = numpy.array(mesh.triangles, dtype=numpy.float32, subok=False)
# tris[..., [1, 2]] = tris[..., [2, 1]]
x = -tris.min(0).min(1)
tris = tris - tris.min(0).min(1)
x = (x / tris.max() + 1 / 16) / (9 / 8)
tris = (tris / tris.max() + 1 / 16) / (9 / 8)

tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')
R = 256
d = torchcumesh2sdf.get_sdf(tris, R, 3 / R).cpu().numpy()
plotlib.ion()
act = plotlib.imshow(d[:, R // 8, :], vmin=-3 / R, vmax=3 / R)
plotlib.colorbar()
plotlib.waitforbuttonpress()
for i in range(0, R):
    if (d[:, i, :] > 1e8).all():
        continue
    act.set_data(d[:, i, :])
    # plotlib.pause(0.08)
    plotlib.waitforbuttonpress()

# print(d)
mcubes.export_obj(*mcubes.marching_cubes(d, 0.88 / R), "tmp/test.obj")
