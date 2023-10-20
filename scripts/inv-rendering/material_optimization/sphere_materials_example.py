import torch
import falcor
import time
import numpy as np
import pyexr as exr
import sys
import os
import dataclasses

sys.path.append("..")
import common
import material_utils
from diff_render_module import DiffRenderModule


@dataclasses.dataclass
class SphereMaterialsExample:
    width: int = 720
    height: int = 432

    M: int = 7                         # Size of the sphere array
    material_count: int = 7 * 7 * 7    # Number of materials
    niter: int = 1000                  # Number of iterations

    spp_ref: int = 2048                # Sample count for the reference images
    spp_primal: int = 32               # Sample count for the primal (forward) pass
    spp_grad: int = 32                 # Sample count for the gradient (backward) pass

    output_dir: str = (                # Output directory
        "../results/sphere_materials/"
    )

    ref_scene_filepath: str = (
        "inv_rendering_scenes/spheres_material_ref.pyscene"
    )

    init_scene_filepath: str = (
        "inv_rendering_scenes/spheres_material_init.pyscene"
    )

    # Initialize material optimization.
    def __post_init__(self):
        # Set up Falcor-Python and render passes.
        self.testbed = common.create_testbed([self.width, self.height])
        device = self.testbed.device
        self.passes = common.create_passes(self.testbed, max_bounces=4, use_war=False)

        # Load the reference scene.
        ref_scene = common.load_scene(
            self.testbed,
            self.ref_scene_filepath,
            self.width / self.height,
        )

        # Set up scene manager and gradients.
        scene_gradients = falcor.SceneGradients.create(
            device=device,
            grad_dim=falcor.uint2(
                self.material_count * falcor.Material.PARAM_COUNT, 0
            ),
            hash_size=falcor.uint2(256, 1),
        )
        self.passes["war_diff_pt"].scene_gradients = scene_gradients

        # Create Falcor buffers for Falcor-PyTorch communication.
        material_ids_buffer = device.create_structured_buffer(
            struct_size=4,
            element_count=self.material_count,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )
        material_params_buffer = device.create_structured_buffer(
            struct_size=4,
            element_count=self.material_count * falcor.Material.PARAM_COUNT,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )
        dL_dI_buffer = device.create_structured_buffer(
            struct_size=12,  # float3
            element_count=self.width * self.height,
            bind_flags=falcor.ResourceBindFlags.ShaderResource
            | falcor.ResourceBindFlags.UnorderedAccess
            | falcor.ResourceBindFlags.Shared,
        )
        buffers = {
            "material_ids": material_ids_buffer,
            "material_params": material_params_buffer,
            "dL_dI": dL_dI_buffer,
        }

        # Set up `material_ids_buffer`.
        material_ids = torch.arange(self.material_count, dtype=torch.int32)
        material_ids_buffer.from_torch(material_ids)
        device.render_context.wait_for_cuda()

        def falcor_to_torch(buffer: falcor.Buffer, dtype=torch.float32):
            params = torch.tensor([0] * buffer.element_count, dtype=dtype)
            buffer.copy_to_torch(params)
            device.render_context.wait_for_cuda()
            return params

        # Get reference material parameters.
        ref_scene.get_material_params(material_ids_buffer, material_params_buffer)
        device.render_context.wait_for_falcor()
        self.ref_raw_material_params = falcor_to_torch(material_params_buffer)
        self.ref_material_params = material_utils.raw_params_to_dicts(
            ref_scene, material_ids, self.ref_raw_material_params
        )

        # Generate reference image.
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
        # Get initial material parameters.
        init_scene.get_material_params(material_ids_buffer, material_params_buffer)
        device.render_context.wait_for_falcor()
        self.init_raw_material_params = falcor_to_torch(material_params_buffer)
        self.init_material_params = material_utils.raw_params_to_dicts(
            init_scene, material_ids, self.init_raw_material_params
        )

        params = {
            "init_material_dicts": self.init_material_params,
            "init_material_raw": self.init_raw_material_params,
            "material_ids": material_ids,
        }

        init_img = common.render_primal(spp["ref"], self.testbed, self.passes)
        print("Output init image with shape", init_img.shape)
        exr.write(os.path.join(self.output_dir, "init.exr"), init_img.cpu().numpy())

        # Set up differentiable render module.
        self.diff_render = DiffRenderModule(
            self.testbed,
            self.passes,
            init_scene,
            params,
            buffers,
            spp,
        )
