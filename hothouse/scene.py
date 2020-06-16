import traitlets
import pythreejs
from IPython.core.display import display

from .model import Model

from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh


class Scene(traitlets.HasTraits):
    components = traitlets.List(trait=traitlets.Instance(Model))
    meshes = traitlets.List()

    def __init__(self):
        self.components = []
        self.meshes = []
        self.embree_scene = rtcs.EmbreeScene()

    def add_component(self, component):
        self.components = self.components + [component]  # Force traitlet update
        self.meshes.append(TriangleMesh(self.embree_scene, component.triangles))


class SceneDisplay(traitlets.HasTraits):

    plant_geom = traitlets.Instance(PlantGeometry)

    def _ipython_display_(self):
        # This needs to actually display, which is not the same as returning a display.
        plantgeometry = pythreejs.BufferGeometry(
            attributes=dict(
                position=pythreejs.BufferAttribute(self.vertices, normalized=False),
                index=pythreejs.BufferAttribute(
                    self.indices.ravel(order="C").astype("u4"), normalized=False
                ),
                color=pythreejs.BufferAttribute(self.colors),
            )
        )
        plantgeometry.exec_three_obj_method("computeFaceNormals")
        mat = pythreejs.MeshStandardMaterial(
            vertexColors="VertexColors", side="DoubleSide"
        )
        myobjectCube = pythreejs.Mesh(
            geometry=plantgeometry, material=mat, position=[0, 0, 0]  # Center the cube
        )
        cCube = pythreejs.PerspectiveCamera(
            position=[25, 35, 100],
            fov=20,
            # children=[pythreejs.DirectionalLight(color='#ffffff', position=[100, 100, 100], intensity=0.5)])
            children=[pythreejs.AmbientLight()],
        )
        sceneCube = pythreejs.Scene(
            children=[myobjectCube, cCube, pythreejs.AmbientLight(color="#dddddd")]
        )

        rendererCube = pythreejs.Renderer(
            camera=cCube,
            background="white",
            background_opacity=1,
            scene=sceneCube,
            controls=[pythreejs.OrbitControls(controlling=cCube)],
            width=800,
            height=800,
        )

        display(rendererCube)
