import numpy as np
import pytz
import datetime

from numpy.testing import assert_almost_equal
from pvlib_model import sun_model

import hothouse
from hothouse.datasets import PLANTS
from hothouse.scene import Scene


def test_sun_blaster():
    ground = np.array([0.0, 0.0, 200.0], dtype="f4")
    up = np.array([0.0, 0.0, 1.0], dtype="f4")
    zenith = 300.0 * up
    north = np.array([0.0, 1.0, 0.0], dtype="f4")
    nx = 512
    ny = 512
    fname = PLANTS.fetch("fullSoy_2-12a.ply")
    p = hothouse.plant_model.PlantModel.from_ply(fname)
    s = Scene(ground=ground)
    s.add_component(p)

    # Champaign
    latitude_deg = 40.1164
    longitude_deg = -88.2434
    tz_champaign = pytz.timezone("America/Chicago")
    date_noon = datetime.datetime(2020, 6, 17, 12, 0, 0, 0, tzinfo=tz_champaign)
    date_sunrise = datetime.datetime(2020, 6, 17, 5, 23, 0, 0, tzinfo=tz_champaign)
    date_sunset = datetime.datetime(2020, 6, 17, 19, 25, 0, 0, tzinfo=tz_champaign)
    date = date_sunrise

    ppfd_tot = sun_model(latitude_deg, longitude_deg, date)  # W m-2
    hits = s.compute_solar_ppfd(
        latitude_deg, longitude_deg, date, ppfd_tot["direct"], ppfd_tot["diffuse"]
    )

    rb = s.get_sun_blaster(latitude_deg, longitude_deg, date, nx=nx, ny=ny)

    o = rb.compute_distance(s)

    assert_almost_equal(rb.center, [408.30853, 183.56668, 577.83765], decimal=5)
    assert_almost_equal(rb.solar_distance, 494.869384765625)
    assert_almost_equal(rb.solar_altitude, 7.8106451271435855)
