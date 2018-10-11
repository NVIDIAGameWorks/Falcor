def render_graph_test():
    skyBox = createRenderPass("SkyBox")

    loadRenderPassLibrary("SamplePassLibrary.Dll")

    test = createRenderGraph()
    test.addPass(createRenderPass("ImageLoader", {"fileName" : "HDRToneMapping.exe.0.png" } ), "ImageLoader")

    test.markOutput("ImageLoader.dst")

    return test

test = render_graph_test()