def render_graph_DefaultRenderGraph():
    g = RenderGraph("DefaultRenderGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("BSDFViewer.dll")
    BSDFViewer = RenderPass("BSDFViewer")
    g.addPass(BSDFViewer, "BSDFViewer")
    AccumulatePass = RenderPass("AccumulatePass", {'enableAccumulation': True, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addEdge("BSDFViewer.output", "AccumulatePass.input")
    g.markOutput("AccumulatePass.output")
    return g

DefaultRenderGraph = render_graph_DefaultRenderGraph()
try: m.addGraph(DefaultRenderGraph)
except NameError: None
