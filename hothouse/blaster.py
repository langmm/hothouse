from enum import Enum
import numpy as np

import pyembree
import traitlets
import traittypes

from .traits_support import check_dtype, check_shape

# pyembree receives origins and directions.


class QueryType(Enum):
    DISTANCE = "DISTANCE"
    OCCLUDED = "OCCLUDED"
    INTERSECT = "INTERSECT"


class RayBlaster(traitlets.HasTraits):
    origins = traittypes.Array().valid(check_shape(None, 3), check_dtype("f4"))
    directions = traittypes.Array().valid(check_shape(None, 3), check_dtype("f4"))

    def cast_once(self, scene, verbose_output=False, query_type=QueryType.DISTANCE):
        output = scene.embree_scene.run(
            self.origins,
            self.directions,
            query=query_type._value_,
            output=verbose_output,
        )
        return output

    def compute_distance(self, scene):
        output = self.cast_once(
            scene, verbose_output=False, query_type=QueryType.DISTANCE
        )
        return output

    def compute_count(self, scene):
        output = self.cast_once(
            scene, verbose_output=True, query_type=QueryType.INTERSECT
        )
        return output


class OrthographicRayBlaster(RayBlaster):
    center = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    forward = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    up = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    east = traittypes.Array().valid(check_dtype("f4"), check_shape(3))
    width = traitlets.CFloat(1.0)
    height = traitlets.CFloat(1.0)
    nx = traitlets.CInt(512)
    ny = traitlets.CInt(512)

    @traitlets.default("east")
    def _default_east(self):
        return np.cross(self.forward, self.up)

    def __init__(self, *args, **kwargs):
        super(OrthographicRayBlaster, self).__init__(*args, **kwargs)

        # here origin is not the center, but the bottom left
        self._directions = np.zeros((self.nx, self.ny, 3), dtype="f4")
        self._directions[:] = self.forward[None, None, :]
        self.directions = self._directions.view().reshape((self.nx * self.ny, 3))

        self._origins = np.zeros((self.nx, self.ny, 3), dtype="f4")
        offset_x, offset_y = np.mgrid[
            -self.width / 2 : self.width / 2 : self.nx * 1j,
            -self.height / 2 : self.height / 2 : self.ny * 1j,
        ]
        self._origins[:] = (
            self.center
            + offset_x[..., None] * self.east
            + offset_y[..., None] * self.up
        )
        self.origins = self._origins.view().reshape((self.nx * self.ny, 3))


class ProjectionRayBlaster(RayBlaster):
    pass


class SphericalRayBlaster(RayBlaster):
    pass
