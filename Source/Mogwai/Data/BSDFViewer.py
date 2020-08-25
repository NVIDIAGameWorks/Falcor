def render_graph_BSDFViewerGraph():
    g = RenderGraph("BSDFViewerGraph")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("BSDFViewer.dll")
    BSDFViewer = createPass("BSDFViewer")
    g.addPass(BSDFViewer, "BSDFViewer")
    AccumulatePass = createPass("AccumulatePass", {'enableAccumulation': True, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addEdge("BSDFViewer.output", "AccumulatePass.input")
    g.markOutput("AccumulatePass.output")
    return g

BSDFViewerGraph = render_graph_BSDFViewerGraph()
try: m.addGraph(BSDFViewerGraph)
except NameError: None
