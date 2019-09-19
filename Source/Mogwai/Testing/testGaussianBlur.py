def render_graph_testGaussianBlur():
    testGaussianBlur = RenderGraph("Gaussian Blur")
    DepthPass = RenderPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
    testGaussianBlur.addPass(DepthPass, "DepthPass")
    SkyBox = RenderPass("SkyBox")
    testGaussianBlur.addPass(SkyBox, "SkyBox")
    ForwardLightingPass = RenderPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
    testGaussianBlur.addPass(ForwardLightingPass, "ForwardLightingPass")
    GaussianBlurPass = RenderPass("GaussianBlurPass")
    testGaussianBlur.addPass(GaussianBlurPass, "GaussianBlur")
    testGaussianBlur.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    testGaussianBlur.addEdge("DepthPass.depth", "SkyBox.depth")
    testGaussianBlur.addEdge("SkyBox.target", "ForwardLightingPass.color")
    testGaussianBlur.addEdge("ForwardLightingPass.color", "GaussianBlur.src")
    testGaussianBlur.markOutput("GaussianBlur.dst")
    return testGaussianBlur

test_Gaussian_Blur = render_graph_testGaussianBlur()
try: m.addGraph(test_Gaussian_Blur)
except NameError: None
