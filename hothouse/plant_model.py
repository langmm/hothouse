from plyfile import PlyData, PlyElement
import numpy as np
from .model import Model

from itertools import tee

# From itertools cookbook
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def _ensure_triangulated(faces):
    for face in faces:
        if len(face[0]) == 3:
            yield face
            continue
        # We are going to make the assumption that the face is convex
        # We choose the first vertex as our fan source
        indices, *rest = face
        base = indices[0]
        for pair in pairwise(indices[1:]):
            yield [np.array((base,) + pair)] + rest



class PlantModel(Model):
    triangles = None
    origin = None
    axial_rotation = None

    def __init__(self, triangles, origin=(0.0, 0.0, 0.0), axial_rotation=0.0,
                 vertices = None, indices = None, colors = None):
        """
        Axial rotation should be in radians
        """
        triangles = triangles + np.array(origin)[None, None, :]  # copy
        if axial_rotation != 0.0:
            # x' = x*cos q - y*sin q
            # y' = x*sin q + y*cos q
            # z' = z
            new_triangles = triangles.copy()
            new_triangles[:,:,0] = (np.cos(axial_rotation) * triangles[:,:,0]
                                    - np.sin(axial_rotation) * triangles[:,:,1])
            new_triangles[:,:,1] = (np.sin(axial_rotation) * triangles[:,:,0]
                                    - np.cos(axial_rotation) * triangles[:,:,1])
            triangles = new_triangles
        self.triangles = triangles
        self.origin = origin
        self.axial_rotation = axial_rotation
        self.vertices = vertices
        self.indices = indices
        self.colors = colors

    @classmethod
    def from_ply(cls, filename, origin=(0.0, 0.0, 0.0), axial_rotation=0.0):
        # This is probably not the absolute best way to do this.
        plydata = PlyData.read(filename)
        vertices = plydata["vertex"][:]
        faces = plydata["face"][:]
        triangles = []
        xyz_faces = []
        for face in _ensure_triangulated(faces):
            indices = face[0]
            vert = vertices[indices]
            triangles.append(np.array([vert["x"], vert["y"], vert["z"]]))
            xyz_faces.append(indices)

        xyz_vert = np.stack([vertices[ax] for ax in 'xyz'], axis=-1)
        xyz_faces = np.concatenate(xyz_faces)
        colors = np.stack([vertices["diffuse_{}".format(c)]
                           for c in ("red", "green", "blue")], axis=-1)
        triangles = np.array(triangles).swapaxes(1, 2)
        obj = cls(triangles, origin, axial_rotation,
                  vertices = xyz_vert, indices = xyz_faces,
                  colors = colors)

        return obj

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

    def _ipython_display_(self):
        # This needs to actually display, which is not the same as returning a display.
        import pythreejs
        from IPython.core.display import display
        plantgeometry = pythreejs.BufferGeometry(attributes=dict(
            position=pythreejs.BufferAttribute(self.vertices, normalized=False),
            index=pythreejs.BufferAttribute(self.indices.ravel(order="C").astype("u4"),
                                            normalized=False),
            color=pythreejs.BufferAttribute(self.colors),
        ))
        plantgeometry.exec_three_obj_method('computeFaceNormals')
        mat = pythreejs.MeshStandardMaterial(vertexColors='VertexColors', side='DoubleSide')
        myobjectCube = pythreejs.Mesh(
            geometry=plantgeometry,
            material=mat,
            position=[0,0,0]   # Center the cube
        )
        cCube = pythreejs.PerspectiveCamera(
            position=[25, 35, 100], fov=20,
            #children=[pythreejs.DirectionalLight(color='#ffffff', position=[100, 100, 100], intensity=0.5)])
            children=[pythreejs.AmbientLight()]
        )
        sceneCube = pythreejs.Scene(children=[myobjectCube, cCube, pythreejs.AmbientLight(color='#dddddd')])

        rendererCube = pythreejs.Renderer(camera=cCube, background='white', background_opacity=1,
                                scene = sceneCube, controls=[pythreejs.OrbitControls(controlling=cCube)],
                                         width=800, height=800)

        display(rendererCube)
