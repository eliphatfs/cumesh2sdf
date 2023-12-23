import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib


mesh = trimesh.load("tmp/spot.obj")
tris = numpy.array(mesh.triangles, dtype=numpy.float32, subok=False)
# tris[..., [1, 2]] = tris[..., [2, 1]]
x = -tris.min(0).min(1)
tris = tris - tris.min(0).min(1)
x = (x / tris.max() + 1 / 16) / (9 / 8)
tris = (tris / tris.max() + 1 / 16) / (9 / 8)

tris = torch.tensor(tris, dtype=torch.float32, device='cuda:0')
R = 64
d = torchcumesh2sdf.get_sdf(tris, R, 4 / R).cpu().numpy()
plotlib.ion()
act = plotlib.imshow(d[:, 0, :], vmin=-0.1, vmax=0.1)
plotlib.colorbar()
plotlib.waitforbuttonpress()
for i in range(R):
    if (d[:, i, :] > 1e8).all():
        continue
    act.set_data(d[:, i, :])
    plotlib.pause(0.08)

# print(d)
mcubes.export_obj(*mcubes.marching_cubes(d, 0), "tmp/test.obj")
