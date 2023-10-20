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

sys.path.append("..")
import common
import material_utils
from loss import compute_render_loss_L1, compute_render_loss_L2
from sphere_materials_example import SphereMaterialsExample


def main(args):
    # Create torch CUDA device
    device = torch.device("cuda:0")
    print(device)
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Create an inverse rendering example that optimizes sphere materials.
    inv_ex = SphereMaterialsExample()

    # Set up PyTorch optimizer.
    learning_rates = {
        falcor.MaterialType.Standard: {
            "base_color": 1e-2,
            "roughness": 3e-3,
            "metallic": 3e-3,
        },
        falcor.MaterialType.PBRTDiffuse: {
            "diffuse": 1e-2,
        },
        falcor.MaterialType.PBRTConductor: {
            "eta": 1e-2,
            "k": 1e-2,
            "roughness": 1e-2,
        },
    }

    params_dicts = inv_ex.init_material_params
    params_list = []
    for material_type in params_dicts:
        for key in params_dicts[material_type]:
            if key == "idx":
                continue
            params_dicts[material_type][key].requires_grad_()
            params_list.append({
                "params": params_dicts[material_type][key],
                "lr": learning_rates[material_type][key],
            })

    optimizer = torch.optim.Adam(params_list, eps=1e-6)

    # Workaround to fix "If capturable=False, state_steps should not be CUDA tensors".
    for param_group in optimizer.param_groups:
        param_group["capturable"] = True

    # Run optimization.
    img_error = []
    param_error = []
    for i_iter in range(inv_ex.niter):
        optimizer.zero_grad()
        loss = 0.0
        now = datetime.datetime.now()

        # Render.
        cur_img = inv_ex.diff_render(
            params_dicts[falcor.MaterialType.Standard]["base_color"],
            params_dicts[falcor.MaterialType.Standard]["metallic"],
            params_dicts[falcor.MaterialType.Standard]["roughness"],
            params_dicts[falcor.MaterialType.PBRTDiffuse]["diffuse"],
            params_dicts[falcor.MaterialType.PBRTConductor]["eta"],
            params_dicts[falcor.MaterialType.PBRTConductor]["k"],
            params_dicts[falcor.MaterialType.PBRTConductor]["roughness"],
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
        params_dicts = material_utils.clamp_material_params(params_dicts)

        param_loss = material_utils.compute_loss_params(params_dicts, inv_ex.ref_material_params)
        param_error.append(param_loss.item())
        print("Parameter loss:", param_loss.item())

        # Export images
        if i_iter % 10 == 0 or i_iter == inv_ex.niter - 1:
            if i_iter <= 50 or i_iter % 50 == 0 or i_iter == inv_ex.niter - 1:
                cur_img = common.render_primal(1024, inv_ex.testbed, inv_ex.passes)
            exr.write(
                os.path.join(inv_ex.output_dir, "iter_{:d}.exr".format(i_iter)),
                cur_img.detach().cpu().numpy(),
            )
            material_utils.output_material_params(
                os.path.join(inv_ex.output_dir, "iter_{:d}.npy".format(i_iter)),
                params_dicts,
            )

    np.savetxt(os.path.join(inv_ex.output_dir, "loss_image.txt"), img_error)
    np.savetxt(os.path.join(inv_ex.output_dir, "loss_parameter.txt"), param_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
