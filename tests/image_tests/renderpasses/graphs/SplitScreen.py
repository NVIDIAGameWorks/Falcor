from falcor import *

def test_SplitScreen():
    imageLoaderA = createPass("ImageLoader", {'filename': 'test_images/cubemap/sorsele3/posz.jpg', 'mips': False, 'srgb': False})
    imageLoaderB = createPass("ImageLoader", {'filename': 'test_images/cubemap/sorsele3/posz.jpg', 'mips': False, 'srgb': True})
    splitScreen = createPass("SplitScreenPass")

    graph = RenderGraph("Split Screen Graph")
    graph.addPass(imageLoaderA, "ImageLoaderA")
    graph.addPass(imageLoaderB, "ImageLoaderB")
    graph.addPass(splitScreen, "SplitScreenPass")

    graph.addEdge("ImageLoaderA.dst", "SplitScreenPass.leftInput")
    graph.addEdge("ImageLoaderB.dst", "SplitScreenPass.rightInput")
    graph.markOutput("SplitScreenPass.output")

    return graph

SplitScreen = test_SplitScreen()
try: m.addGraph(SplitScreen)
except NameError: None
