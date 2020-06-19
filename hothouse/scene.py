import traittypes
import traitlets
import pythreejs
import numpy as np
from IPython.core.display import display

from .model import Model
from .blaster import RayBlaster, SunRayBlaster
from .traits_support import check_shape, check_dtype

from pyembree import rtcore_scene as rtcs
from pyembree.mesh_construction import TriangleMesh


class Scene(traitlets.HasTraits):
    
    ground = traittypes.Array(np.array([0.0, 0.0, 0.0], "f4")).valid(check_dtype("f4"), check_shape(3))
    up = traittypes.Array(np.array([0.0, 0.0, 1.0], "f4")).valid(
        check_dtype("f4"), check_shape(3))
    north = traittypes.Array(np.array([0.0, 1.0, 0.0], "f4")).valid(
        check_dtype("f4"), check_shape(3))
    components = traitlets.List(trait=traitlets.Instance(Model))
    blasters = traitlets.List(trait=traitlets.Instance(RayBlaster))
    meshes = traitlets.List(trait=traitlets.Instance(TriangleMesh))
    embree_scene = traitlets.Instance(rtcs.EmbreeScene, args=tuple())

    # TODO: Add surface for ground so that reflection from ground
    # is taken into account

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

    def get_sun_blaster(self, latitude, longitude, date, **kwargs):
        r"""Get a sun blaster that is adjusted for this scene so that
        the blaster will never intercept a component in the scene. This
        distance is determined by computing the maximum distance of any
        vertex in the scene from the ground parameter.

        Args:
            latitude (float): Latitude (in degrees) of the scene.
            longitude (float): Longitude (in degrees) of the scene.
            date (datetime.datetime): Time when PPFD should be calculated.
                This determines the incidence angle of light from the
                sun.

        Returns:
            SunRayBlaster: Blaster tuned to this scene.

        """
        max_distance2 = 0.0
        for c in self.components:
            max_distance2 = max(
                max_distance2,
                np.max(np.sum((c.vertices-self.ground)**2, axis=1)))
        max_distance = np.sqrt(max_distance2)
        kwargs.setdefault('zenith', self.up * max_distance)
        kwargs.setdefault('width', 2 * max_distance)
        kwargs.setdefault('height', 2 * max_distance)
        blaster = SunRayBlaster(latitude=latitude,
                                longitude=longitude, date=date,
                                ground=self.ground, north=self.north,
                                **kwargs)
        return blaster

    def compute_solar_ppfd(self, latitude, longitude, date,
                           direct_ppfd, diffuse_ppfd, **kwargs):
        r"""Compute the photosynthetic photon flux density (PPFD) on
        each scene element from the sun.

        Args:
            latitude (float): Latitude (in degrees) of the scene.
            longitude (float): Longitude (in degrees) of the scene.
            date (datetime.datetime): Time when PPFD should be calculated.
                This determines the incidence angle of light from the
                sun.
            direct_ppfd (float): Direct Photosynthetic Photon Flux
                Density (PPFD) at the surface of the Earth for the
                specified location and time.
            diffuse_ppfd (float): Diffuse Photosynthetic Photon Flux
                Density (PPFD) at the surface of the Earth for the
                specified location and time.

        Returns:
            dict: Mapping from scene component to an array of PPFD
                values for each triangle in the component.

        """
        blaster = self.get_sun_blaster(latitude, longitude, date)
        counts = self.compute_hit_count(blaster)
        component_ppfd = {}
        direct_ppfd_per_ray = (
            direct_ppfd * blaster.width * blaster.height
            / (blaster.nx * blaster.ny))
        for ci, component in enumerate(self.components):
            norms = component.norms
            areas = component.areas
            aoi = np.arccos(
                np.dot(norms, -blaster.forward)
                / (2.0 * areas * np.linalg.norm(blaster.forward)))
            aoi[aoi > np.pi/2] -= np.pi  # Side of leaf independent
            ppfd = counts[ci] * direct_ppfd_per_ray * np.cos(aoi) / areas
            component_ppfd[ci] = ppfd
        return component_ppfd
        

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
