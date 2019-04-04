from plyfile import PlyData, PlyElement
import numpy as np

class PlantModel:
    triangles = None
    origin = None
    axial_rotation = None

    def __init__(self, triangles, origin = (0.0, 0.0, 0.0), axial_rotation = 0.0):
        self.triangles = triangles # not copying
        self.origin = origin
        self.axial_rotation = axial_rotation

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
            triangles.append(np.array(vert['x'], vert['y'], vert['z']))
        triangles = np.array(triangles).swapaxes(0,2)
        obj = cls(triangles, origin, axial_rotation)
        return obj
