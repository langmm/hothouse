import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
fname = PLANTS.fetch('fullSoy_2-12a.ply')
p = hothouse.plant_model.PlantModel.from_ply(fname)
scene = Scene(ground=ground)
scene.add_component(p)


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
date = date_sunrise


def plot_light(scene, latitude_deg, longitude_deg, date,
               fname="distance.png"):
    # Solar radiation model including atmosphere
    ppfd_tot = sun_model(latitude_deg, longitude_deg, date)  # W m-2


    # Blaster representing the sun
    nx = ny = 1024
    sun = scene.get_sun_blaster(latitude_deg, longitude_deg, date,
                                nx=nx, ny=ny, direct_ppfd=ppfd_tot['direct'],
                                diffuse_ppfd=ppfd_tot['diffuse'])


    # Camera
    center = np.array([0.0, -100.0, 500], dtype='f4')
    forward = np.array([0.0, 1.0, 0.0], dtype='f4')
    up = np.array([0.0, 0.0, 1.0], dtype='f4')
    nx = ny = sun.nx
    width = height = 800
    camera = OrthographicRayBlaster(center=center, forward=forward, up=up,
                                    width=width, height=height,
                                    nx=nx, ny=ny)

    # Compute flux density on scene from sun
    o = camera.compute_flux_density(scene, sun)
    o[o <= 0] = np.nan

    # Plot result
    plt.clf()
    plt.imshow(o.reshape((camera.ny, camera.nx), order='F'),
               origin='lower', norm=LogNorm(50, 5.0e4))
    plt.colorbar()
    if fname is not None:
        fname = plt.savefig(fname)
    else:
        plt.show()


plot_light(scene, latitude_deg, longitude_deg, date)


# for hr in np.linspace(5.3833, 19.4166, 10):
#     date = datetime.datetime(2020, 6, 17, int(hr), int((60 * hr) % 60), 0, 0,
#                              tzinfo=tz_champaign)
#     plot_light(scene, latitude_deg, longitude_deg, date, fname=None)
