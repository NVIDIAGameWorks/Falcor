from falcor import *

def render_graph_MegakernelPathTracerVBuffer():
    g = RenderGraph("MegakernelPathTracerVBuffer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("MegakernelPathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    VBufferRT = createPass("VBufferRT")
    g.addPass(VBufferRT, "VBufferRT")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'operator': ToneMapOp.Linear, 'clamp': False, 'outputFormat': ResourceFormat.RGBA32Float})
    g.addPass(ToneMapper, "ToneMapper")
    AccumulatePass = createPass("AccumulatePass")
    g.addPass(AccumulatePass, "AccumulatePass")
    MegakernelPathTracer = createPass("MegakernelPathTracer", {'mSharedParams': PathTracerParams(useVBuffer=1)})
    g.addPass(MegakernelPathTracer, "MegakernelPathTracer")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("VBufferRT.vbuffer", "MegakernelPathTracer.vbuffer")
    g.addEdge("MegakernelPathTracer.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

MegakernelPathTracerVBuffer = render_graph_MegakernelPathTracerVBuffer()
try: m.addGraph(MegakernelPathTracerVBuffer)

except NameError: None
