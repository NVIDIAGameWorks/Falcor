from falcor import *

def render_graph_ForwardRendering():
    loadRenderPassLibrary("DepthPass.dll")
    loadRenderPassLibrary("ForwardLightingPass.dll")
    loadRenderPassLibrary("BlitPass.dll")
    testForwardRendering = RenderGraph("ForwardRenderer")
    DepthPass = createPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
    testForwardRendering.addPass(DepthPass, "DepthPass")
    SkyBox = createPass("SkyBox")
    testForwardRendering.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = createPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
    testForwardRendering.addPass(ForwardLightingPass, "ForwardLightingPass")
    BlitPass = createPass("BlitPass", {'filter': SamplerFilter.Linear})
    testForwardRendering.addPass(BlitPass, "BlitPass")
    testForwardRendering.addEdge("ForwardLightingPass.color", "BlitPass.src")
    testForwardRendering.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testForwardRendering.addEdge("DepthPass.depth", "SkyBox.depth")
    testForwardRendering.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testForwardRendering.markOutput("BlitPass.dst")
    testForwardRendering.markOutput("ForwardLightingPass.motionVecs")
    return testForwardRendering

ForwardRendering = render_graph_ForwardRendering()
try: m.addGraph(ForwardRendering)
except NameError: None
