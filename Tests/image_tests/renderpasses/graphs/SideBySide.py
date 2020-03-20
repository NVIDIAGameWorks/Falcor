from falcor import *

def test_SideBySide():
    loadRenderPassLibrary("DebugPasses.dll")
    imageLoaderA = RenderPass("ImageLoader", {'filename': 'Cubemaps\\Sorsele3\\posz.jpg', 'mips': False, 'srgb': False})
    imageLoaderB = RenderPass("ImageLoader", {'filename': 'Cubemaps\\Sorsele3\\posz.jpg', 'mips': False, 'srgb': True})
    sideComparison = RenderPass("SideBySidePass")

    graph = RenderGraph("Side by Side")
    graph.addPass(imageLoaderA, "ImageLoaderA")
    graph.addPass(imageLoaderB, "ImageLoaderB")
    graph.addPass(sideComparison, "SideBySidePass")

    graph.addEdge("ImageLoaderA.dst", "SideBySidePass.leftInput")
    graph.addEdge("ImageLoaderB.dst", "SideBySidePass.rightInput")
    graph.markOutput("SideBySidePass.output")

    return graph

SideBySide = test_SideBySide()
try: m.addGraph(SideBySide)
except NameError: None
