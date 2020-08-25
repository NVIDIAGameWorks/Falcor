from falcor import *

def render_graph_MegakernelPathTracerGBuffer():
    g = RenderGraph("MegakernelPathTracerGBuffer")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("MegakernelPathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    GBufferRT = createPass("GBufferRT")
    g.addPass(GBufferRT, "GBufferRT")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'operator': ToneMapOp.Linear, 'clamp': False, 'outputFormat': ResourceFormat.RGBA32Float})
    g.addPass(ToneMapper, "ToneMapper")
    AccumulatePass = createPass("AccumulatePass")
    g.addPass(AccumulatePass, "AccumulatePass")
    MegakernelPathTracer = createPass("MegakernelPathTracer", {'mSharedParams': PathTracerParams(useVBuffer=0)})
    g.addPass(MegakernelPathTracer, "MegakernelPathTracer")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.addEdge("GBufferRT.posW", "MegakernelPathTracer.posW")
    g.addEdge("GBufferRT.normW", "MegakernelPathTracer.normalW")
    g.addEdge("GBufferRT.tangentW", "MegakernelPathTracer.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "MegakernelPathTracer.faceNormalW")
    g.addEdge("GBufferRT.viewW", "MegakernelPathTracer.viewW")
    g.addEdge("GBufferRT.diffuseOpacity", "MegakernelPathTracer.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "MegakernelPathTracer.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "MegakernelPathTracer.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "MegakernelPathTracer.mtlParams")
    g.addEdge("MegakernelPathTracer.color", "AccumulatePass.input")
    g.markOutput("ToneMapper.dst")
    return g

MegakernelPathTracerGBuffer = render_graph_MegakernelPathTracerGBuffer()
try: m.addGraph(MegakernelPathTracerGBuffer)

except NameError: None
