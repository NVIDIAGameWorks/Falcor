import torch
import falcor
import time
import numpy as np
import pyexr as exr
import sys
import os
import dataclasses

CUR_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CUR_DIR, ".."))
import common
import mesh_utils
from diff_render_module import DiffRenderModule


@dataclasses.dataclass
class ShapeExample:
    width: int = 512
    height: int = 512

    mesh_id: int = 6                   # Indicate which mesh to optimize
    max_bounces: int = 0               # Max number of indirect bounces

    niter: int = 400                   # Number of iterations
    lr: float = 0.01                   # Learning rate
    lambda_value: float = 60.0         # Lambda value for the largesteps optimizer

    spp_ref: int = 256                 # Sample count for the reference images
    spp_primal: int = 32               # Sample count for the primal (forward) pass
    spp_grad: int = 32                 # Sample count for the gradient (backward) pass

    output_dir: str = (                # Output directory
        CUR_DIR + "/results/"
    )

    ref_scene_filepath: str = (
        "inv_rendering_scenes/bunny_ref.pyscene"
    )

    init_scene_filepath: str = (
        "inv_rendering_scenes/bunny_init.pyscene"
    )

    # Initialize material optimization.
    def __post_init__(self):
        # Set up Falcor-Python and render passes.
        self.testbed = common.create_testbed([self.width, self.height])
        device = self.testbed.device
        self.passes = common.create_passes(self.testbed, self.max_bounces, use_war=True)

        # Load the reference scene.
        ref_scene = common.load_scene(
            self.testbed,
            self.ref_scene_filepath,
            self.width / self.height,
        )

        # Render the reference image.
        spp = {
            "ref": self.spp_ref,
            "primal": self.spp_primal,
            "grad": self.spp_grad,
        }
        self.ref_img = common.render_primal(spp["ref"], self.testbed, self.passes)
        print("Output ref image with shape", self.ref_img.shape)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        exr.write(os.path.join(self.output_dir, "ref.exr"), self.ref_img.cpu().numpy())

        # Load the initial scene.
        init_scene = common.load_scene(
            self.testbed,
            self.init_scene_filepath,
            self.width / self.height,
        )

        # Load the mesh.
        self.mesh = mesh_utils.Mesh();
        self.mesh.load_from_falcor(self.testbed, self.mesh_id)
        mesh_vertex_count = self.mesh.v_pos.shape[0]

        self.params = {
            "mesh_id": self.mesh_id,
            "mesh_positions": self.mesh.v_pos.detach().clone(),
        }

        # Set up mesh to optimize.
        self.passes["war_diff_pt"].set_mesh_to_optimize(self.mesh_id)

        # Set up scene gradients.
        scene_gradients = falcor.SceneGradients.create(
            device=device,
            grad_config_list=[
                falcor.GradConfig(
                    grad_type=falcor.GradientType.MeshPosition,
                    dim=mesh_vertex_count * 3,
                    hash_size=128,
                ),
                falcor.GradConfig(
                    grad_type=falcor.GradientType.MeshNormal,
                    dim=mesh_vertex_count * 3,
                    hash_size=128,
                ),
                falcor.GradConfig(
                    grad_type=falcor.GradientType.MeshTangent,
                    dim=mesh_vertex_count * 3,
                    hash_size=128,
                ),
            ],
        )
        self.passes["war_diff_pt"].scene_gradients = scene_gradients

        # Create Falcor buffers for Falcor-PyTorch communication.
        dL_dI_buffer = device.create_structured_buffer(
            struct_size=12,  # float3
            element_count=self.width * self.height,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )
        buffers = {
            "dL_dI": dL_dI_buffer,
        }

        init_img = common.render_primal(spp["ref"], self.testbed, self.passes)
        print("Output init image with shape", init_img.shape)
        exr.write(os.path.join(self.output_dir, "init.exr"), init_img.cpu().numpy())

        # Set up differentiable render module.
        self.diff_render = DiffRenderModule(
            self.testbed,
            self.passes,
            init_scene,
            self.mesh,
            self.params,
            buffers,
            spp,
        )
