from falcor import *

def render_graph_CompositePass():
    loadRenderPassLibrary("Utils.dll")
    g = RenderGraph("Composite Pass")
    Composite = RenderPass("Composite", {'scaleA': 1.5, 'scaleB': 1.0})
    g.addPass(Composite, "Composite")
    ImageLoader = RenderPass("ImageLoader", {'filename': 'Cubemaps\\Sorsele3\\posz.jpg', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoader, "ImageLoader")
    ImageLoader_ = RenderPass("ImageLoader", {'filename': 'smoke-puff.png', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoader_, "ImageLoader_")
    g.addEdge("ImageLoader.dst", "Composite.A")
    g.addEdge("ImageLoader_.dst", "Composite.B")
    g.markOutput("Composite.out")
    return g

CompositePass = render_graph_CompositePass()
try: m.addGraph(CompositePass)
except NameError: None
