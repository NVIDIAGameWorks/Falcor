import torch
import falcor
import time
import numpy as np
import pyexr as exr
import sys
import os
import dataclasses
import datetime
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import common
from largesteps_optimizer import LargeSteps
from loss import compute_render_loss_L2
from shape_example import ShapeExample


def main(args):
    # Create torch CUDA device
    device = torch.device("cuda:0")
    print(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Create an inverse rendering example that optimizes sphere materials.
    inv_ex = ShapeExample(
        init_scene_filepath="inv_rendering_scenes/sphere_init.pyscene",
        niter=1000,
        max_bounces=2,
    )

    # Set up largesteps optimizer.
    mesh_positions = inv_ex.params["mesh_positions"].clone().requires_grad_(True)
    tri_idx = inv_ex.mesh.tri_idx
    optimizer = LargeSteps([mesh_positions], tri_idx, inv_ex.lr, (0.9, 0.999), inv_ex.lambda_value)

    # Run optimization.
    img_error = []
    for i_iter in range(inv_ex.niter):
        optimizer.zero_grad()
        loss = 0.0
        now = datetime.datetime.now()

        # Compute mesh positions, normals, and tangents.
        inv_ex.mesh.v_pos = mesh_positions
        mesh_normals, mesh_tangents = inv_ex.mesh.compute_shading_frame()

        # Render.
        cur_img = inv_ex.diff_render(
            mesh_positions,
            mesh_normals,
            mesh_tangents,
        )
        cur_img[cur_img.isnan()] = 0.0

        img_loss = compute_render_loss_L2(cur_img, inv_ex.ref_img)
        loss += img_loss.item()
        img_error.append(img_loss.item())

        # Backpropagate gradients.
        img_loss.backward()

        end = datetime.datetime.now() - now
        time_iter = end.seconds + end.microseconds / 1e6

        # Print stats
        print(
            "[INFO] iter = {:d}, image loss = {:.3f}, time = {:.3f}s".format(
                i_iter, loss, time_iter
            )
        )

        optimizer.step()

        # Export images
        if i_iter % 10 == 0 or i_iter == inv_ex.niter - 1:
            if i_iter <= 50 or i_iter % 50 == 0 or i_iter == inv_ex.niter - 1:
                cur_img = common.render_primal(256, inv_ex.testbed, inv_ex.passes)
            exr.write(
                os.path.join(inv_ex.output_dir, "iter_{:d}.exr".format(i_iter)),
                cur_img.detach().cpu().numpy(),
            )

    np.savetxt(os.path.join(inv_ex.output_dir, "loss_image.txt"), img_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
