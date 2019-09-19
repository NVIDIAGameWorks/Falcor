def render_graph_testFXAA():
    testFXAA = RenderGraph("ForwardRenderer")
    DepthPass = RenderPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
    testFXAA.addPass(DepthPass, "DepthPass")
    SkyBox = RenderPass("SkyBox")
    testFXAA.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = RenderPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
    testFXAA.addPass(ForwardLightingPass, "ForwardLightingPass")
    FXAAPass = RenderPass("FXAAPass")
    testFXAA.addPass(FXAAPass, "FXAA")
    testFXAA.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testFXAA.addEdge("DepthPass.depth", "SkyBox.depth")
    testFXAA.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testFXAA.addEdge("ForwardLightingPass.color", "FXAA.src")
    testFXAA.markOutput("FXAA.dst")
    return testFXAA

test_FXAA = render_graph_testFXAA()
try: m.addGraph(test_FXAA)
except NameError: None
