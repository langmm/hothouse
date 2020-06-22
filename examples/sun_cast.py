import matplotlib.pyplot as plt
import numpy as np
import datetime
import pytz
import hothouse
from hothouse.datasets import PLANTS
from hothouse.scene import Scene
from hothouse.blaster import OrthographicRayBlaster
from pvlib_model import sun_model


# Create the scene
ground = np.array([0.0, 0.0, 200.0], dtype='f4')
up = np.array([0.0, 0.0, 1.0], dtype='f4')
zenith = 300.0 * up
north = np.array([0.0, 1.0, 0.0], dtype='f4')
nx = 512
ny = 512
fname = PLANTS.fetch('fullSoy_2-12a.ply')
p = hothouse.plant_model.PlantModel.from_ply(fname)
s = Scene(ground=ground)
s.add_component(p)


# Location specifics for Champaign
latitude_deg = 40.1164
longitude_deg = -88.2434
tz_champaign = pytz.timezone("America/Chicago")
date_noon = datetime.datetime(2020, 6, 17, 12, 0, 0, 0,
                              tzinfo=tz_champaign)
date_sunrise = datetime.datetime(2020, 6, 17, 5, 23, 0, 0,
                                 tzinfo=tz_champaign)
date_sunset = datetime.datetime(2020, 6, 17, 19, 25, 0, 0,
                                tzinfo=tz_champaign)
date = date_noon


# Solar radiation model including atmosphere
ppfd_tot = sun_model(latitude_deg, longitude_deg, date)  # W m-2


# Blaster representing the sun
sun = s.get_sun_blaster(latitude_deg, longitude_deg, date,
                        nx=nx, ny=ny, direct_ppfd=ppfd_tot['direct'],
                        diffuse_ppfd=ppfd_tot['diffuse'])


# Camera
center = np.array([0.0, -100.0, 200], dtype='f4')
forward = np.array([0.0, 1.0, 0.0], dtype='f4')
up = np.array([0.0, 0.0, 1.0], dtype='f4')
nx = ny = 1024
width = 512
height = 512
camera = OrthographicRayBlaster(center=center, forward=forward, up=up,
                                width=width, height=height,
                                nx=nx, ny=ny)

# Compute flux density on scene from sun
o = camera.compute_flux_density(s, sun)
o[o == 0] = np.nan


# Plot result
plt.clf()
plt.imshow(o.reshape((ny, nx), order='F'), origin='lower')
plt.colorbar()
plt.savefig("distance.png")
