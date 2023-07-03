from falcor import *

def render_graph_CrossFadePass():
    g = RenderGraph("CrossFade Pass")
    CrossFade = createPass("CrossFade")
    g.addPass(CrossFade, "CrossFade")
    ImageLoaderA = createPass("ImageLoader", {'filename': 'test_images/cubemap/sorsele3/posz.jpg', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoaderA, "ImageLoaderA")
    ImageLoaderB = createPass("ImageLoader", {'filename': 'test_images/smoke_puff.png', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoaderB, "ImageLoaderB")
    g.addEdge("ImageLoaderA.dst", "CrossFade.A")
    g.addEdge("ImageLoaderB.dst", "CrossFade.B")
    g.markOutput("CrossFade.out")
    return g

CrossFadePass = render_graph_CrossFadePass()
try: m.addGraph(CrossFadePass)
except NameError: None
