from falcor import *

def render_graph_CompositePass():
    g = RenderGraph("Composite Pass")
    Composite = createPass("Composite")
    g.addPass(Composite, "Composite")
    ImageLoaderA = createPass("ImageLoader", {'filename': 'test_images/cubemap/sorsele3/posz.jpg', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoaderA, "ImageLoaderA")
    ImageLoaderB = createPass("ImageLoader", {'filename': 'test_images/smoke_puff.png', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoaderB, "ImageLoaderB")
    g.addEdge("ImageLoaderA.dst", "Composite.A")
    g.addEdge("ImageLoaderB.dst", "Composite.B")
    g.markOutput("Composite.out")
    return g

CompositePass = render_graph_CompositePass()
try: m.addGraph(CompositePass)
except NameError: None
