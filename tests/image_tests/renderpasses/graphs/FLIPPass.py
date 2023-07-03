from falcor import *

def test_FLIPPass():
    imageLoaderA = createPass("ImageLoader", {'filename': 'test_images/cubemap/sorsele3/posz.jpg', 'mips': False, 'srgb': False})
    imageLoaderB = createPass("ImageLoader", {'filename': 'test_images/cubemap/sorsele3/posz.jpg', 'mips': False, 'srgb': True})
    flip = createPass("FLIPPass")

    graph = RenderGraph("FLIP")
    graph.addPass(imageLoaderA, "ImageLoaderA")
    graph.addPass(imageLoaderB, "ImageLoaderB")
    graph.addPass(flip, "FLIP")
    graph.addEdge("ImageLoaderA.dst", "FLIP.referenceImage")
    graph.addEdge("ImageLoaderB.dst", "FLIP.testImage")
    graph.markOutput("FLIP.errorMapDisplay")

    return graph

FLIPPass = test_FLIPPass()
try: m.addGraph(FLIPPass)
except NameError: None
