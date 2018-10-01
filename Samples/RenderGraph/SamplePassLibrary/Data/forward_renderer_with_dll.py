def render_graph_forward_renderer():
    skyBox = createRenderPass("SkyBox")

    loadRenderPassLibrary("SamplePassLibrary.Dll")
    
    forward_renderer = createRenderGraph()
    forward_renderer.addPass(createRenderPass("DepthPass"), "DepthPrePass")
    forward_renderer.addPass(createRenderPass("ForwardLightingPass"), "LightingPass")
    forward_renderer.addPass(createRenderPass("CascadedShadowMaps"), "ShadowPass")
    forward_renderer.addPass(createRenderPass("MyBlitPass"), "MyBlitPass")
    forward_renderer.addPass(createRenderPass("ToneMapping"), "ToneMapping")
    forward_renderer.addPass(createRenderPass("SSAO"), "SSAO")
    forward_renderer.addPass(createRenderPass("FXAA"), "FXAA")

    forward_renderer.addPass(skyBox, "SkyBox")

    forward_renderer.addEdge("DepthPrePass.depth", "SkyBox.depth");
    forward_renderer.addEdge("SkyBox.target", "LightingPass.color");
    forward_renderer.addEdge("DepthPrePass.depth", "ShadowPass.depth");
    forward_renderer.addEdge("DepthPrePass.depth", "LightingPass.depth");
    forward_renderer.addEdge("ShadowPass.visibility", "LightingPass.visibilityBuffer");
    forward_renderer.addEdge("LightingPass.color", "ToneMapping.src");
    forward_renderer.addEdge("ToneMapping.dst", "SSAO.colorIn");
    forward_renderer.addEdge("LightingPass.normals", "SSAO.normals");
    forward_renderer.addEdge("LightingPass.depth", "SSAO.depth");
    forward_renderer.addEdge("SSAO.colorOut", "FXAA.src");
    forward_renderer.addEdge("FXAA.dst", "MyBlitPass.src");

    forward_renderer.markOutput("MyBlitPass.dst")
    return forward_renderer

forward_renderer = render_graph_forward_renderer()