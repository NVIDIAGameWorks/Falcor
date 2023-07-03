from falcor import *

def render_graph_MinimalPathTracer():
    g = RenderGraph("MinimalPathTracer")
    MinimalPathTracer = createPass("MinimalPathTracer")
    g.addPass(MinimalPathTracer, "MinimalPathTracer")
    VBufferRT = createPass("VBufferRT")
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass")
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'operator': 'Linear', 'clamp': False, 'outputFormat': 'RGBA32Float'})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
    g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
    g.addEdge("MinimalPathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    g.markOutput("ToneMapper.dst", TextureChannelFlags.Alpha)
    return g

MinimalPathTracer = render_graph_MinimalPathTracer()
try: m.addGraph(MinimalPathTracer)
except NameError: None
