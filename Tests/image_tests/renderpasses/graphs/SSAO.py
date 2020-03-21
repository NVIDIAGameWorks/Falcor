from falcor import *

def render_graph_SSAO():
    loadRenderPassLibrary("DepthPass.dll")
    loadRenderPassLibrary("ForwardLightingPass.dll")
    testSSAO = RenderGraph("ForwardRenderer")
    DepthPass = RenderPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
    testSSAO.addPass(DepthPass, "DepthPass")
    SkyBox = RenderPass("SkyBox")
    testSSAO.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = RenderPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
    testSSAO.addPass(ForwardLightingPass, "ForwardLightingPass")
    SSAOPass = RenderPass("SSAO")
    testSSAO.addPass(SSAOPass, "SSAO")
    testSSAO.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testSSAO.addEdge("DepthPass.depth", "SkyBox.depth")
    testSSAO.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testSSAO.addEdge("DepthPass.depth", "SSAO.depth")
    testSSAO.addEdge("ForwardLightingPass.color", "SSAO.colorIn")
    testSSAO.markOutput("SSAO.colorOut")
    return testSSAO

SSAO = render_graph_SSAO()
try: m.addGraph(SSAO)
except NameError: None
