def render_graph_testTAA():
    testTAA = RenderGraph("ForwardRenderer")
    DepthPass = RenderPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
    testTAA.addPass(DepthPass, "DepthPass")
    SkyBox = RenderPass("SkyBox")
    testTAA.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = RenderPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
    testTAA.addPass(ForwardLightingPass, "ForwardLightingPass")
    TAAPass = RenderPass("TemporalAAPass")
    testTAA.addPass(TAAPass, "TAA")
    testTAA.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testTAA.addEdge("DepthPass.depth", "SkyBox.depth")
    testTAA.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testTAA.addEdge("ForwardLightingPass.color", "TAA.colorIn")
    testTAA.addEdge("ForwardLightingPass.motionVecs", "TAA.motionVecs")
    testTAA.markOutput("TAA.colorOut")
    return testTAA

test_TAA = render_graph_testTAA()
try: m.addGraph(test_TAA)
except NameError: None
