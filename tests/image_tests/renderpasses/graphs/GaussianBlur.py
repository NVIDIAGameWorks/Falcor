from falcor import *

def render_graph_GaussianBlur():
    testGaussianBlur = RenderGraph("Gaussian Blur")
    imageLoader = createPass("ImageLoader", {'filename' : "test_scenes/envmaps/hallstatt4_hd.hdr", 'mips': False, 'srgb': False, 'outputFormat': 'RGBA32Float'})
    testGaussianBlur.addPass(imageLoader, "ImageLoader")
    GaussianBlurPass = createPass("GaussianBlur")
    testGaussianBlur.addPass(GaussianBlurPass, "GaussianBlur")
    testGaussianBlur.addEdge("ImageLoader.dst", "GaussianBlur.src")
    testGaussianBlur.markOutput("GaussianBlur.dst")
    return testGaussianBlur

GaussianBlur = render_graph_GaussianBlur()
try: m.addGraph(GaussianBlur)
except NameError: None
