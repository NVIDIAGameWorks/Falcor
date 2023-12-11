import torch
import falcor

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common
import mesh_utils


class DiffRenderFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mesh_positions,
        mesh_normals,
        mesh_tangents,
        context,
    ):
        ctx.context = context

        testbed : falcor.Testbed = context["testbed"]
        passes : dict[str, falcor.RenderPass] = context["passes"]
        mesh : mesh_utils.Mesh = context["mesh"]
        params = context["params"]

        # Update mesh.
        mesh.v_pos = mesh_positions.detach()
        mesh.v_normal = mesh_normals.detach()
        mesh.v_tangent = mesh_tangents.detach()
        mesh.update_to_falcor(testbed, params["mesh_id"])

        # Falcor forward rendering.
        img = common.render_primal(context["spp"]["primal"], testbed, passes)
        return img.detach()

    @staticmethod
    def backward(ctx, grad_output):
        context = ctx.context
        testbed : falcor.Testbed = context["testbed"]
        passes : dict[str, falcor.RenderPass] = context["passes"]
        dL_dI_buffer : falcor.Buffer = context["buffers"]["dL_dI"]

        # Set `dL_dI_buffer`.
        dL_dI_buffer.from_torch(grad_output)
        testbed.device.render_context.wait_for_cuda()

        # Falcor differentiable rendering.
        grad_raw = common.render_grad(
            context["spp"]["grad"],
            testbed,
            passes,
            dL_dI_buffer,
        )
        grad_position = grad_raw[falcor.GradientType.MeshPosition].view(-1, 3)
        grad_normal = grad_raw[falcor.GradientType.MeshNormal].view(-1, 3)
        grad_tangent = grad_raw[falcor.GradientType.MeshTangent].view(-1, 3)

        return (
            grad_position,
            grad_normal,
            grad_tangent,
            None,
        )


class DiffRenderModule(torch.nn.Module):
    def __init__(
        self,
        testbed,
        passes,
        scene,
        mesh,
        params,
        buffers,
        spp,
    ):
        super(DiffRenderModule, self).__init__()
        self.context = dict(
            testbed=testbed,
            passes=passes,
            scene=scene,
            mesh=mesh,
            params=params,
            buffers=buffers,
            spp=spp,
        )

    def forward(
        self,
        mesh_positions,
        mesh_normals,
        mesh_tangents,
    ):
        return DiffRenderFunction.apply(
            mesh_positions,
            mesh_normals,
            mesh_tangents,
            self.context,
        )
