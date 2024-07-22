import time
import diso
import numpy
import torch
import mcubes
import trimesh
import torchcumesh2sdf
import matplotlib.pyplot as plotlib


R = 256
band = 1 / R


def load_and_preprocess(p):
    mesh = trimesh.load(p)
    tris = numpy.array(mesh.triangles, dtype=numpy.float32, subok=False)
    # tris[..., [1, 2]] = tris[..., [2, 1]]
    tris = tris - tris.min(0).min(0)
    tris = (tris / tris.max() + band) / (1 + band * 2)
    return torch.tensor(tris, dtype=torch.float32, device='cuda:0')


s = [load_and_preprocess(r"untitled.glb")] * 1
for tris in s:
    for i in range(100):
        start = time.perf_counter()
        d = torchcumesh2sdf.get_udf(tris, R, band)
        print("%.3f ms" % ((time.perf_counter() - start) * 1000))

v, t = diso.DiffMC().cuda().forward(d)
d = d.cpu().numpy()
cc = torchcumesh2sdf.get_collide(tris, R, band)
numpy.savez_compressed("test.npz", sdf=d.astype(numpy.float16), rays=cc.cpu().numpy())
# print((d < 1e8).sum())
# breakpoint()
# d = numpy.load("test.npz")['sdf'].astype(numpy.float32)


# visualize
plotlib.ion()
act = plotlib.imshow(d[:, 0, :], vmin=-3 / R, vmax=3 / R)
plotlib.colorbar()
for i in range(0, R):
    if (d[:, i, :] > 1e8).all():
        continue
    print(i)
    act.set_data(d[:, i, :])
    # plotlib.pause(0.01)
    plotlib.waitforbuttonpress()

# run marching cubes to reconstruct
mcubes.export_obj(v.cpu().numpy(), t.cpu().numpy(), "tmp/test.obj")
