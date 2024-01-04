import time
import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib


R = 512
band = 3 / R


def load_and_preprocess(p):
    mesh = trimesh.load(p)
    tris = numpy.array(mesh.triangles, dtype=numpy.float32, subok=False)
    # tris[..., [1, 2]] = tris[..., [2, 1]]
    tris = tris - tris.min(0).min(0)
    tris = (tris / tris.max() + band) / (1 + band * 2)
    return torch.tensor(tris, dtype=torch.float32, device='cuda:0')


s = [load_and_preprocess("tmp/Octocat-v2.obj")] * 10
for tris in s:
    start = time.perf_counter()
    d = torchcumesh2sdf.get_sdf(tris, R, band)
    torch.cuda.synchronize()
    print("%.3f ms" % ((time.perf_counter() - start) * 1000))

d = d.cpu().numpy()

# visualize
plotlib.ion()
act = plotlib.imshow(d[:, 0, :], vmin=-3 / R, vmax=3 / R)
plotlib.colorbar()
for i in range(0, R):
    if (d[:, i, :] > 1e8).all():
        continue
    act.set_data(d[:, i, :])
    plotlib.pause(0.01)
    # plotlib.waitforbuttonpress()

# run marching cubes to reconstruct
mcubes.export_obj(*mcubes.marching_cubes(d, 0 * 0.87 / R), "tmp/test.obj")
