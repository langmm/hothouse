import traitlets
import pythreejs
import numpy as np
from IPython.core.display import display

from .model import Model
from .blaster import RayBlaster

from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh


class Scene(traitlets.HasTraits):
    components = traitlets.List(trait=traitlets.Instance(Model))
    blasters = traitlets.List(trait=traitlets.Instance(RayBlaster))
    meshes = traitlets.List(trait=traitlets.Instance(TriangleMesh))
    embree_scene = traitlets.Instance(rtcs.EmbreeScene)

    def __init__(self):
        self.components = []
        self.meshes = []
        self.embree_scene = rtcs.EmbreeScene()

    def add_component(self, component):
        self.components = self.components + [component]  # Force traitlet update
        self.meshes.append(TriangleMesh(self.embree_scene, component.triangles))

    def compute_hit_count(self, blaster):
        output = blaster.compute_count(self)
        component_counts = {}
        for ci, component in enumerate(self.components):
            hits = output["primID"][output["geomID"] == ci]
            component_counts[ci] = np.bincount(
                hits[hits >= 0], minlength=component.triangles.shape[0]
            )
        return component_counts

    def _ipython_display_(self):
        # This needs to actually display, which is not the same as returning a display.
        cam = pythreejs.PerspectiveCamera(
            position=[25, 35, 100], fov=20, children=[pythreejs.AmbientLight()],
        )
        children = [cam, pythreejs.AmbientLight(color="#dddddd")]
        material = pythreejs.MeshBasicMaterial(
            color="#ff0000", vertexColors="VertexColors", side="DoubleSide"
        )
        for model in self.components:
            mesh = pythreejs.Mesh(
                geometry=model.geometry, material=material, position=[0, 0, 0]
            )
            children.append(mesh)

        scene = pythreejs.Scene(children=children)

        rendererCube = pythreejs.Renderer(
            camera=cam,
            background="white",
            background_opacity=1,
            scene=scene,
            controls=[pythreejs.OrbitControls(controlling=cam)],
            width=800,
            height=800,
        )

        return rendererCube
