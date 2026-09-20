from falcor import *

def render_graph_PathTracerReSTIRPT():
    g = RenderGraph("PathTracerReSTIRPT")
    PathTracer = createPass("PathTracer", {
        'samplesPerPixel': 1,
        'tracePassMode': 'MinimalPayload',
        'useReSTIRPT': True,
        'emissiveSampler': 'Power',
        'maxDiffuseBounces': 10,
        'maxSpecularBounces': 10,
        'useRussianRoulette': True,
        'usePairedNeighbors': True,
        'reSTIRPathTracingOptions': {'useAreaReservoirs': False}
    })
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Center', 'sampleCount': 1, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

PathTracerReSTIRPT = render_graph_PathTracerReSTIRPT()
try: m.addGraph(PathTracerReSTIRPT)
except NameError: None
