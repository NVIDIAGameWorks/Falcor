def render_graph_forward_renderer():
    skyBox = RenderPass("SkyBox")
    
    forward_renderer = RenderGraph("ForwardRenderer")
    forward_renderer.addPass(RenderPass("DepthPass"), "DepthPrePass")
    forward_renderer.addPass(RenderPass("ForwardLightingPass"), "LightingPass")
    forward_renderer.addPass(RenderPass("CascadedShadowMaps"), "ShadowPass")
    forward_renderer.addPass(RenderPass("BlitPass"), "BlitPass")
    forward_renderer.addPass(RenderPass("ToneMappingPass"), "ToneMapping")
    forward_renderer.addPass(RenderPass("SSAOPass"), "SSAO")
    forward_renderer.addPass(RenderPass("FXAAPass"), "FXAA")

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
    forward_renderer.addEdge("FXAA.dst", "BlitPass.src");

    forward_renderer.markOutput("BlitPass.dst")
    
    return forward_renderer

forward_renderer = render_graph_forward_renderer()
try: m.addGraph(forward_renderer)
except NameError: None
