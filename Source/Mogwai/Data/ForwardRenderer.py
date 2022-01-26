def render_graph_ForwardRenderer():
    loadRenderPassLibrary("Antialiasing.dll")
    loadRenderPassLibrary("BlitPass.dll")
    loadRenderPassLibrary("CSM.dll")
    loadRenderPassLibrary("DepthPass.dll")
    loadRenderPassLibrary("ForwardLightingPass.dll")
    loadRenderPassLibrary("SSAO.dll")
    loadRenderPassLibrary("ToneMapper.dll")

    ForwardRenderer = RenderGraph("ForwardRenderer")

    ForwardRenderer.addPass(createPass("DepthPass"), "DepthPrePass")
    ForwardRenderer.addPass(createPass("ForwardLightingPass"), "LightingPass")
    ForwardRenderer.addPass(createPass("CSM"), "ShadowPass")
    ForwardRenderer.addPass(createPass("BlitPass"), "BlitPass")
    ForwardRenderer.addPass(createPass("ToneMapper", {'autoExposure': True}), "ToneMapping")
    ForwardRenderer.addPass(createPass("SSAO"), "SSAO")
    ForwardRenderer.addPass(createPass("FXAA"), "FXAA")
    ForwardRenderer.addPass(createPass("SkyBox"), "SkyBox")

    ForwardRenderer.addEdge("DepthPrePass.depth", "SkyBox.depth")
    ForwardRenderer.addEdge("SkyBox.target", "LightingPass.color")
    ForwardRenderer.addEdge("DepthPrePass.depth", "ShadowPass.depth")
    ForwardRenderer.addEdge("DepthPrePass.depth", "LightingPass.depth")
    ForwardRenderer.addEdge("ShadowPass.visibility", "LightingPass.visibilityBuffer")
    ForwardRenderer.addEdge("LightingPass.color", "ToneMapping.src")
    ForwardRenderer.addEdge("ToneMapping.dst", "SSAO.colorIn")
    ForwardRenderer.addEdge("LightingPass.normals", "SSAO.normals")
    ForwardRenderer.addEdge("LightingPass.depth", "SSAO.depth")
    ForwardRenderer.addEdge("SSAO.colorOut", "FXAA.src")
    ForwardRenderer.addEdge("FXAA.dst", "BlitPass.src")

    ForwardRenderer.markOutput("BlitPass.dst")

    return ForwardRenderer

ForwardRenderer = render_graph_ForwardRenderer()
try: m.addGraph(ForwardRenderer)
except NameError: None
