from falcor import *

def render_graph_WARDiffPathTracer():
    g = RenderGraph("WARDiffPathTracer")
    WARDiffPathTracer = createPass("WARDiffPathTracer", {"maxBounces": 0, "samplesPerPixel": 1, "diffMode": "BackwardDiffDebug", "diffVarName": "CBOX_BUNNY_TRANSLATION"})
    g.addPass(WARDiffPathTracer, "WARDiffPathTracer")

    AccumulatePassPrimal = createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"})
    g.addPass(AccumulatePassPrimal, "AccumulatePassPrimal")

    AccumulatePassDiff = createPass("AccumulatePass", {"enabled": True, 'precisionMode': "Single"})
    g.addPass(AccumulatePassDiff, "AccumulatePassDiff")
    ColorMapPassDiff = createPass("ColorMapPass", {"minValue": -4.0, "maxValue": 4.0, "autoRange": False})
    g.addPass(ColorMapPassDiff, "ColorMapPassDiff")

    g.addEdge("WARDiffPathTracer.color", "AccumulatePassPrimal.input")
    g.addEdge("WARDiffPathTracer.dColor", "AccumulatePassDiff.input")
    g.addEdge("AccumulatePassDiff.output", "ColorMapPassDiff.input")

    g.markOutput("AccumulatePassDiff.output")
    g.markOutput("ColorMapPassDiff.output")
    return g

WARDiffPathTracer = render_graph_WARDiffPathTracer()
try: m.addGraph(WARDiffPathTracer)
except NameError: None
