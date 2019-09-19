def test_split_screen():
    loadRenderPassLibrary("DebugPasses.dll")
    imageLoaderA = RenderPass("ImageLoader", {'filename': 'Cubemaps\\Sorsele3\\posz.jpg', 'mips': False, 'srgb': False})
    imageLoaderB = RenderPass("ImageLoader", {'filename': 'Cubemaps\\Sorsele3\\posz.jpg', 'mips': False, 'srgb': True})
    splitScreen = RenderPass("SplitScreenPass")

    graph = RenderGraph("Split Screen Graph")
    graph.addPass(imageLoaderA, "ImageLoaderA")
    graph.addPass(imageLoaderB, "ImageLoaderB")
    graph.addPass(splitScreen, "SplitScreenPass")

    graph.addEdge("ImageLoaderA.dst", "SplitScreenPass.leftInput")
    graph.addEdge("ImageLoaderB.dst", "SplitScreenPass.rightInput")
    graph.markOutput("SplitScreenPass.output")

    return graph

split_screen_graph = test_split_screen()

m.addGraph(split_screen_graph)
