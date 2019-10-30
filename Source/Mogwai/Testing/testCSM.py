def render_graph_testCSM():
    csm = RenderGraph("Cascaded Shadow Maps")
    csm.addPass(RenderPass("DepthPass"), "DepthPrePass")
    csm.addPass(RenderPass("CascadedShadowMaps"), "ShadowPass")

    csm.addEdge("DepthPrePass.depth", "ShadowPass.depth");

    csm.markOutput("ShadowPass.visibility")
    
    return csm

csm = render_graph_testCSM()
try: m.addGraph(csm)
except NameError: None
