from falcor import *

def render_graph_GaussianBlur():
    loadRenderPassLibrary("DepthPass.dll")
    loadRenderPassLibrary("ForwardLightingPass.dll")
    loadRenderPassLibrary("Utils.dll")
    testGaussianBlur = RenderGraph("Gaussian Blur")
    DepthPass = RenderPass("DepthPass")
    testGaussianBlur.addPass(DepthPass, "DepthPass")
    SkyBox = RenderPass("SkyBox")
    testGaussianBlur.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = RenderPass("ForwardLightingPass")
    testGaussianBlur.addPass(ForwardLightingPass, "ForwardLightingPass")
    GaussianBlurPass = RenderPass("GaussianBlur")
    testGaussianBlur.addPass(GaussianBlurPass, "GaussianBlur")
    testGaussianBlur.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testGaussianBlur.addEdge("DepthPass.depth", "SkyBox.depth")
    testGaussianBlur.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testGaussianBlur.addEdge("ForwardLightingPass.color", "GaussianBlur.src")
    testGaussianBlur.markOutput("GaussianBlur.dst")
    return testGaussianBlur

GaussianBlur = render_graph_GaussianBlur()
try: m.addGraph(GaussianBlur)
except NameError: None
