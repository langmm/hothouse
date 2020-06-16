import traitlets
from plyfile import PlyData, PlyElement
import numpy as np
from .model import Model


class PlantModel(Model):
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
