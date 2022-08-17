from falcor import *

def render_graph_FXAA():
    loadRenderPassLibrary("Antialiasing.dll")
    loadRenderPassLibrary("DepthPass.dll")
    loadRenderPassLibrary("ForwardLightingPass.dll")
    testFXAA = RenderGraph("ForwardRenderer")
    DepthPass = createPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
    testFXAA.addPass(DepthPass, "DepthPass")
    SkyBox = createPass("SkyBox")
    testFXAA.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = createPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
    testFXAA.addPass(ForwardLightingPass, "ForwardLightingPass")
    FXAAPass = createPass("FXAA")
    testFXAA.addPass(FXAAPass, "FXAA")
    testFXAA.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testFXAA.addEdge("DepthPass.depth", "SkyBox.depth")
    testFXAA.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testFXAA.addEdge("ForwardLightingPass.color", "FXAA.src")
    testFXAA.markOutput("FXAA.dst")
    return testFXAA

FXAA = render_graph_FXAA()
try: m.addGraph(FXAA)
except NameError: None
