import traitlets
from plyfile import PlyData, PlyElement
import numpy as np
from .model import Model


class PlantModel(Model):
    triangles = None
    origin = None
    axial_rotation = None

    def __init__(
        self,
        triangles,
        origin=(0.0, 0.0, 0.0),
        axial_rotation=0.0,
        vertices=None,
        indices=None,
        colors=None,
    ):
        """
        Axial rotation should be in radians
        """
        triangles = triangles + np.array(origin)[None, None, :]  # copy
        if axial_rotation != 0.0:
            # x' = x*cos q - y*sin q
            # y' = x*sin q + y*cos q
            # z' = z
            new_triangles = triangles.copy()
            new_triangles[:, :, 0] = (
                np.cos(axial_rotation) * triangles[:, :, 0]
                - np.sin(axial_rotation) * triangles[:, :, 1]
            )
            new_triangles[:, :, 1] = (
                np.sin(axial_rotation) * triangles[:, :, 0]
                - np.cos(axial_rotation) * triangles[:, :, 1]
            )
            triangles = new_triangles
        self.triangles = triangles
        self.origin = origin
        self.axial_rotation = axial_rotation
        self.vertices = vertices
        self.indices = indices
        self.colors = colors

    def clone(self, origin=(0.0, 0.0, 0.0), axial_rotation=0.0):
        """
        This will clone this plant, but with a new origin and a new axial rotation.
        Note that because this applies a re-centering, not a translation, this
        is likely better used on the original object rather than objects that
        have already been translated or rotated.
        """
        return PlantModel(
            self.triangles.copy(), origin=origin, axial_rotation=axial_rotation
        )
