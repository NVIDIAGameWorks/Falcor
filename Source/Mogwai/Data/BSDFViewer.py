def render_graph_BSDFViewer():
    g = RenderGraph("BSDFViewer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("BSDFViewer.dll")
    BSDFViewer = createPass("BSDFViewer", {'materialID': 0})
    g.addPass(BSDFViewer, "BSDFViewer")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': AccumulatePrecision.Double})
    g.addPass(AccumulatePass, "AccumulatePass")
    g.addEdge("BSDFViewer.output", "AccumulatePass.input")
    g.markOutput("AccumulatePass.output")
    return g

BSDFViewer = render_graph_BSDFViewer()
try: m.addGraph(BSDFViewer)
except NameError: None
