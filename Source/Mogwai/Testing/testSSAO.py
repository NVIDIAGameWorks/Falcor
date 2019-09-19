def render_graph_testSSAO():
    testSSAO = RenderGraph("ForwardRenderer")
    DepthPass = RenderPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
    testSSAO.addPass(DepthPass, "DepthPass")
    SkyBox = RenderPass("SkyBox")
    testSSAO.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = RenderPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
    testSSAO.addPass(ForwardLightingPass, "ForwardLightingPass")
    SSAOPass = RenderPass("SSAOPass")
    testSSAO.addPass(SSAOPass, "SSAO")
    testSSAO.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testSSAO.addEdge("DepthPass.depth", "SkyBox.depth")
    testSSAO.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testSSAO.addEdge("DepthPass.depth", "SSAO.depth")
    testSSAO.addEdge("ForwardLightingPass.color", "SSAO.colorIn")
    testSSAO.markOutput("SSAO.colorOut")
    return testSSAO

test_SSAO = render_graph_testSSAO()
try: m.addGraph(test_SSAO)
except NameError: None
