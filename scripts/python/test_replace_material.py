import falcor

def setup_renderpass(testbed):
    render_graph = testbed.create_render_graph("PathTracer")
    render_graph.create_pass("PathTracer", "PathTracer", {'samplesPerPixel': 1})
    render_graph.create_pass("VBufferRT", "VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
    render_graph.create_pass("AccumulatePass", "AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    render_graph.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    render_graph.add_edge("PathTracer.color", "AccumulatePass.input")
    render_graph.mark_output("AccumulatePass.output")
    testbed.render_graph = render_graph

def main():
    falcor.Logger.verbosity = falcor.Logger.Level.Info

    scene_path = 'test_scenes/cornell_box.pyscene'

    # Create device and setup renderer.
    device = falcor.Device(type=falcor.DeviceType.D3D12, gpu=0, enable_debug_layer=False)
    testbed = falcor.Testbed(width=1920, height=1080, create_window=True, device=device)
    setup_renderpass(testbed)

    # Load scene.
    testbed.load_scene(scene_path)
    testbed.frame()

    # Create replacement materials.
    mat1 = falcor.PBRTDiffuseMaterial(device, "PBRT diffuse")
    mat1.load_texture(falcor.MaterialTextureSlot.BaseColor, 'test_scenes/textures/checker_tile_base_color.png')
    mat2 = falcor.NeuralMaterial(device, "Neural material", "test_scenes/materials/neural/material_4.json")

    # Replace materials.
    print('Replacing materials ...')
    testbed.scene.replace_material(0, mat1)
    testbed.scene.replace_material(1, mat2)

    testbed.run()


if __name__ == "__main__":
    main()
