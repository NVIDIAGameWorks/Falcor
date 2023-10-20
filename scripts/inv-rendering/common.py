import os
import sys
import argparse
import signal
import sys
from glob import glob
from pathlib import Path
import torch
import falcor
import numpy as np
import pyexr as exr
import json
from PIL import Image
import io


def load_scene(testbed: falcor.Testbed, scene_path: Path, aspect_ratio=1.0):
    flags = (
        falcor.SceneBuilderFlags.DontMergeMaterials
        | falcor.SceneBuilderFlags.RTDontMergeDynamic
        | falcor.SceneBuilderFlags.DontOptimizeMaterials
    )
    testbed.load_scene(scene_path, flags)
    testbed.scene.camera.aspectRatio = aspect_ratio
    testbed.scene.renderSettings.useAnalyticLights = False
    testbed.scene.renderSettings.useEnvLight = False
    return testbed.scene


def create_testbed(reso: (int, int)):
    device_id = 0
    testbed = falcor.Testbed(
        width=reso[0], height=reso[1], create_window=True, gpu=device_id
    )
    testbed.show_ui = False
    testbed.clock.time = 0
    testbed.clock.pause()
    return testbed


def create_passes(testbed: falcor.Testbed, max_bounces: int, use_war: bool):
    # Rendering graph of the WAR differentiable path tracer.
    render_graph = testbed.create_render_graph("WARDiffPathTracer")
    primal_accumulate_pass = render_graph.create_pass(
        "PrimalAccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    grad_accumulate_pass = render_graph.create_pass(
        "GradAccumulatePass",
        "AccumulatePass",
        {"enabled": True, "precisionMode": "Single"},
    )
    war_diff_pt_pass = render_graph.create_pass(
        "WARDiffPathTracer",
        "WARDiffPathTracer",
        {
            "samplesPerPixel": 1,
            "maxBounces": max_bounces,
            "diffMode": "BackwardDiff",
            "useWAR": use_war,
        },
    )
    render_graph.add_edge("WARDiffPathTracer.color", "PrimalAccumulatePass.input")
    render_graph.add_edge("WARDiffPathTracer.dColor", "GradAccumulatePass.input")
    render_graph.mark_output("PrimalAccumulatePass.output")
    render_graph.mark_output("GradAccumulatePass.output")

    passes = {
        "primal_accumulate": primal_accumulate_pass,
        "grad_accumulate": grad_accumulate_pass,
        "war_diff_pt": war_diff_pt_pass,
    }

    testbed.render_graph = render_graph
    return passes


def render_primal(spp: int, testbed: falcor.Testbed, passes):
    passes["war_diff_pt"].run_backward = 0
    passes["primal_accumulate"].reset()
    for i in range(spp):
        testbed.frame()

    img = testbed.render_graph.get_output("PrimalAccumulatePass.output").to_numpy()
    img = torch.from_numpy(img[:, :, :3]).cuda()
    return img


def render_grad(spp: int, testbed: falcor.Testbed, passes, dL_dI_buffer, grad_type):
    passes["war_diff_pt"].run_backward = 1
    passes["war_diff_pt"].dL_dI = dL_dI_buffer

    scene_gradients = passes["war_diff_pt"].scene_gradients
    scene_gradients.clear(testbed.device.render_context, grad_type)

    for _ in range(spp):
        testbed.frame()

    scene_gradients.aggregate(testbed.device.render_context, grad_type)

    grad_buffer = scene_gradients.get_grads_buffer(grad_type)
    grad = torch.tensor([0] * (grad_buffer.size // 4), dtype=torch.float32)
    grad_buffer.copy_to_torch(grad)
    testbed.device.render_context.wait_for_cuda()
    return grad / float(spp)
