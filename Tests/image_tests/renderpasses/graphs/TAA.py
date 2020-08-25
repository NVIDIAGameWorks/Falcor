from falcor import *

def render_graph_TAA():
    loadRenderPassLibrary("Antialiasing.dll")
    loadRenderPassLibrary("DepthPass.dll")
    loadRenderPassLibrary("ForwardLightingPass.dll")
    testTAA = RenderGraph("TAA")
    DepthPass = createPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
    testTAA.addPass(DepthPass, "DepthPass")
    SkyBox = createPass("SkyBox")
    testTAA.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = createPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
    testTAA.addPass(ForwardLightingPass, "ForwardLightingPass")
    TAAPass = createPass("TAA")
    testTAA.addPass(TAAPass, "TAA")
    testTAA.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testTAA.addEdge("DepthPass.depth", "SkyBox.depth")
    testTAA.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testTAA.addEdge("ForwardLightingPass.color", "TAA.colorIn")
    testTAA.addEdge("ForwardLightingPass.motionVecs", "TAA.motionVecs")
    testTAA.markOutput("TAA.colorOut")
    return testTAA

TAA = render_graph_TAA()
try: m.addGraph(TAA)
except NameError: None
