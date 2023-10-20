import torch
import falcor

import sys
sys.path.append("..")
import common
import material_utils


class DiffRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        standard_base_color,
        standard_metallic,
        standard_roughness,
        diffuse_color,
        conductor_eta,
        conductor_k,
        conductor_roughness,
        context,
    ):
        ctx.context = context

        testbed : falcor.Testbed = context["testbed"]
        passes : dict[str, falcor.RenderPass] = context["passes"]
        scene : falcor.Scene  = context["scene"]
        params = context["params"]
        material_params_dicts = params["init_material_dicts"]

        # Update material parameters.
        new_dicts = {
            falcor.MaterialType.Standard: {
                "base_color": standard_base_color.detach(),
                "metallic": standard_metallic.detach(),
                "roughness": standard_roughness.detach(),
                "idx": material_params_dicts[falcor.MaterialType.Standard]["idx"],
            },
            falcor.MaterialType.PBRTDiffuse: {
                "diffuse": diffuse_color.detach(),
                "idx": material_params_dicts[falcor.MaterialType.PBRTDiffuse]["idx"],
            },
            falcor.MaterialType.PBRTConductor: {
                "eta": conductor_eta.detach(),
                "k": conductor_k.detach(),
                "roughness": conductor_roughness.detach(),
                "idx": material_params_dicts[falcor.MaterialType.PBRTConductor]["idx"],
            },
        }

        # Convert material parameter dictionaries to flattened raw parameters for Falcor.
        material_params_raw = material_utils.dicts_to_raw_params(
            testbed.scene, params["material_ids"], new_dicts, params["init_material_raw"]
        )

        material_ids_buffer : falcor.Buffer = context["buffers"]["material_ids"]
        material_params_buffer : falcor.Buffer = context["buffers"]["material_params"]

        # Set material parameters for Falcor.
        material_params_buffer.from_torch(material_params_raw)
        testbed.device.render_context.wait_for_cuda()
        scene.set_material_params(material_ids_buffer, material_params_buffer)

        # Falcor forward rendering.
        img = common.render_primal(context["spp"]["primal"], testbed, passes)
        return img.detach()

    @staticmethod
    def backward(ctx, grad_output):
        context = ctx.context
        testbed : falcor.Testbed = context["testbed"]
        passes = context["passes"]
        dL_dI_buffer = context["buffers"]["dL_dI"]

        # Set `dL_dI_buffer`.
        dL_dI_buffer.from_torch(grad_output)
        testbed.device.render_context.wait_for_cuda()

        # Falcor differentiable rendering.
        grad_raw = common.render_grad(
            context["spp"]["grad"],
            testbed,
            passes,
            dL_dI_buffer,
            falcor.GradientType.Material,
        )
        grad = material_utils.raw_params_to_dicts(testbed.scene, context["params"]["material_ids"], grad_raw)

        return (
            grad[falcor.MaterialType.Standard]["base_color"],
            grad[falcor.MaterialType.Standard]["metallic"],
            grad[falcor.MaterialType.Standard]["roughness"],
            grad[falcor.MaterialType.PBRTDiffuse]["diffuse"],
            grad[falcor.MaterialType.PBRTConductor]["eta"],
            grad[falcor.MaterialType.PBRTConductor]["k"],
            grad[falcor.MaterialType.PBRTConductor]["roughness"],
            None,
        )


class DiffRenderModule(torch.nn.Module):
    def __init__(
        self,
        testbed,
        passes,
        scene,
        params,
        buffers,
        spp,
    ):
        super(DiffRenderModule, self).__init__()
        self.context = dict(
            testbed=testbed,
            passes=passes,
            scene=scene,
            params=params,
            buffers=buffers,
            spp=spp,
        )

    def forward(
        self,
        standard_base_color,
        standard_metallic,
        standard_roughness,
        diffuse_color,
        conductor_eta,
        conductor_k,
        conductor_roughness,
    ):
        return DiffRenderFunction.apply(
            standard_base_color,
            standard_metallic,
            standard_roughness,
            diffuse_color,
            conductor_eta,
            conductor_k,
            conductor_roughness,
            self.context,
        )
