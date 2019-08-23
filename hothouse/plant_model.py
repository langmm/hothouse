from plyfile import PlyData, PlyElement
import numpy as np
from .model import Model

class PlantModel(Model):
    triangles = None
    origin = None
    axial_rotation = None

    def __init__(self, triangles, origin = (0.0, 0.0, 0.0), axial_rotation = 0.0):
        self.triangles = triangles + np.array(origin)[None, :, None] # copy
        self.origin = origin
        self.axial_rotation = axial_rotation
        # TODO: implement axial rotation

    @classmethod
    def from_ply(cls, filename, origin = (0.0, 0.0, 0.0), axial_rotation = 0.0):
        # This is probably not the absolute best way to do this.
        plydata = PlyData.read(filename)
        vertices = plydata['vertex'][:]
        faces = plydata['face'][:]
        triangles = []
        for face in faces:
            indices = face[0]
            vert = vertices[indices]
            triangles.append(np.array([vert['x'], vert['y'], vert['z']]))
        triangles = np.array(triangles).swapaxes(1,2)
        obj = cls(triangles, origin, axial_rotation)
        return obj

    def clone(self, origin = (0.0, 0.0, 0.0), axial_rotation = 0.0):
        """
        This will clone this plant, but with a new origin and a new axial rotation.
        Note that because this applies a re-centering, not a translation, this
        is likely better used on the original object rather than objects that
        have already been translated or rotated.
        """
        return PlantModel(self.triangles.copy(),
                          origin = origin,
                          axial_rotation = axial_rotation)
