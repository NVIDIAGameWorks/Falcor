def render_graph_test():
    loadRenderPassLibrary("SamplePassLibrary.Dll")

    test = createRenderGraph()
    test.addPass(createRenderPass("ImageLoader", {"fileName" : "HDRToneMapping.exe.0.png" } ), "ImageLoader")
    test.addPass(createRenderPass("MyBlitPass"), "MyBlitPass")

    test.addEdge("ImageLoader.dst", "MyBlitPass.src");

    test.markOutput("MyBlitPass.dst")

    return test



def render_graph_blit_test():
    
    test = createRenderGraph()
    test.addPass(createRenderPass("ImageLoader", {"fileName" : "HDRToneMapping.exe.0.png" } ), "ImageLoader")
    test.addPass(createRenderPass("BlitPass"), "BlitPass")

    test.addEdge("ImageLoader.dst", "BlitPass.src");

    test.markOutput("BlitPass.dst")

    return test