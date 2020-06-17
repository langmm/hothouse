import matplotlib.pyplot as plt
import numpy as np
import datetime
import pytz
import hothouse
from hothouse.datasets import PLANTS
from hothouse.scene import Scene

fname = PLANTS.fetch('fullSoy_2-12a.ply')
p = hothouse.plant_model.PlantModel.from_ply(fname)
s = Scene()
s.add_component(p)

# Champaign
latitude_deg = 40.1164
longitude_deg = -88.2434
tz_champaign = pytz.timezone("America/Chicago")
date_noon = datetime.datetime(2020, 6, 17, 12, 0, 0, 0,
                              tzinfo=tz_champaign)
date_sunrise = datetime.datetime(2020, 6, 17, 5, 23, 0, 0,
                                 tzinfo=tz_champaign)
date_sunset = datetime.datetime(2020, 6, 17, 19, 25, 0, 0,
                                tzinfo=tz_champaign)

date = date_sunrise
ground = np.array([0.0, 0.0, 200.0], dtype='f4')
zenith = np.array([0.0, 0.0, 300.0], dtype='f4')
north = np.array([0.0, 1.0, 0.0], dtype='f4')
nx = 512
ny = 512
width = 1024
height = 1024
rb = hothouse.SunRayBlaster(latitude=latitude_deg,
                            longitude=longitude_deg,
                            date=date,
                            ground=ground,
                            zenith=zenith,
                            north=north,
                            nx=nx, ny=ny,
                            width=width, height=height)

o = rb.compute_distance(s)
# o = rb.cast_once(s)
o[o > 1e35] = np.nan

plt.clf()
plt.imshow(o.reshape((ny, nx), order='F'), origin='lower')
plt.colorbar()
plt.savefig("distance.png")


# output = s.compute_hit_count(rb)
# import pdb; pdb.set_trace()
