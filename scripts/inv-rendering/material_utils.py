import torch
import falcor
import numpy as np

from loss import compute_render_loss_L2


def raw_params_to_dicts(scene : falcor.Scene, material_ids, raw_params):
    """
    @param[in] scene: falcor.Scene.
    @param[in] material_ids: torch.IntTensor[M], material ids.
    @param[in] raw_params: torch.FloatTensor[20 * M], flattened material parameters from Falcor.
    @return parameter dictionaries.
    """
    device = torch.device("cuda:0")
    params_dicts = {
        falcor.MaterialType.Standard: {
            "base_color": torch.zeros(0, device=device),
            "metallic": torch.zeros(0, device=device),
            "roughness": torch.zeros(0, device=device),
            "idx": [],
        },
        falcor.MaterialType.PBRTDiffuse: {
            "diffuse": torch.zeros(0, device=device),
            "idx": [],
        },
        falcor.MaterialType.PBRTConductor: {
            "eta": torch.zeros(0, device=device),
            "k": torch.zeros(0, device=device),
            "roughness": torch.zeros(0, device=device),
            "idx": [],
        }
    }

    material_param_size = raw_params.shape[0] // material_ids.shape[0]
    for i in range(material_ids.shape[0]):
        material_type = scene.get_material(material_ids[i]).type
        params = params_dicts[material_type]
        layout = falcor.get_material_param_layout(material_type)
        for key in params:
            if key == "idx":
                params[key].append(i)
            else:
                offset = layout[key]["offset"]
                size = layout[key]["size"]
                params[key] = torch.concat(
                    (
                        params[key],
                        raw_params[i * material_param_size + offset : i * material_param_size + offset + size],
                    )
                )

    return params_dicts


def dicts_to_raw_params(scene, material_ids, params_dicts, input_raw_params):
    """
    @param[in] scene: falcor.Scene.
    @param[in] material_ids: torch.IntTensor[M], material ids.
    @param[in] params_dicts: parameter dictionaries.
    @param[in] input_raw_params: torch.FloatTensor[20 * M], original flattened material parameters.
    @return updated flattened material parameters for Falcor
    """
    material_parameter_count = input_raw_params.shape[0] // material_ids.shape[0]
    res = input_raw_params.clone()
    for material_type in params_dicts:
        layout = falcor.get_material_param_layout(material_type)
        params = params_dicts[material_type]
        for i in range(len(params["idx"])):
            idx = params["idx"][i]
            for key in params:
                if key == "idx":
                    continue
                offset = layout[key]["offset"]
                size = layout[key]["size"]
                res[
                    idx * material_parameter_count + offset : idx * material_parameter_count + offset + size
                ] = params[key][i * size : i * size + size]

    return res


def compute_loss_params(params_dicts, ref_params_dicts):
    res = 0.0
    for material_type in params_dicts:
        params = params_dicts[material_type]
        for key in params:
            if key == "idx":
                continue
            res += compute_render_loss_L2(
                params[key].detach(),
                ref_params_dicts[material_type][key],
            )
    return res


def output_material_params(filename, params_dicts):
    res = {}
    for material_type in params_dicts:
        if material_type == falcor.MaterialType.Standard:
            type_name = "Standard"
        elif material_type == falcor.MaterialType.PBRTDiffuse:
            type_name = "PBRTDiffuse"
        elif material_type == falcor.MaterialType.PBRTConductor:
            type_name = "PBRTConductor"
        else:
            raise RuntimeError("Unknown material type")

        res[type_name] = {}
        params = params_dicts[material_type]
        for key in params:
            if key == "idx":
                res[type_name][key] = np.array(params[key], dtype=np.int32)
            else:
                res[type_name][key] = params[key].detach().cpu().numpy()
    np.save(filename, res)


def clamp_material_params(params_dicts):
    for material_type in params_dicts:
        params = params_dicts[material_type]
        if len(params["idx"]) == 0:
            continue

        for key in params:
            if key == "idx":
                continue
            elif key == "roughness":
                params[key].data.copy_(torch.clamp(params[key], 0.1, 0.9999))
            else:
                params[key].data.copy_(torch.clamp(params[key], 0.0001, 0.9999))

    return params_dicts
