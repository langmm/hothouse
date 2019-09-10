import matplotlib.pyplot as plt
import hothouse
import numpy as np
from hothouse.datasets import PLANTS

#center = np.array([0.0, -100.0, 200])
#forward = np.array([0.0, 1.0, 0.0])
#up = np.array([0.0, 0.0, 1.0])
#Npix = 1024
#width = 400
#height = 400

def make_image(in_buf):
    # center, forward, up, Npix, width, height
    if not isinstance(in_buf, dict):
        raise RuntimeError
    center = np.array(in_buf['center'])
    forward = np.array(in_buf['forward'])
    up = np.array(in_buf['up'])
    Npix = in_buf['Npix']
    width = in_buf['width']
    height = in_buf['height']

    rb = hothouse.OrthographicRayBlaster(center, forward, up,
                                         width, height, Npix, Npix)

    fname = PLANTS.fetch('fullSoy_2-12a.ply')
    p1 = hothouse.PlantModel.from_ply(fname)
    p2 = hothouse.PlantModel.from_ply(fname, origin = (100,100,0))

    s = hothouse.Scene()
    s.add_component(p1)
    s.add_component(p2)

    o = rb.cast_once(s)
    o[o > 1e35] = np.nan

    return o.reshape((Npix, Npix), order='F')
