from falcor import *

def render_graph_PathTracerMaterials():
    g = RenderGraph("PathTracerMaterials")
    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("PathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")

    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1, 'maxSurfaceBounces': 3})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': AccumulatePrecision.Single})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    # Final frame output
    g.markOutput("ToneMapper.dst")
    g.markOutput("ToneMapper.dst", TextureChannelFlags.Alpha)

    return g

PathTracerMaterials = render_graph_PathTracerMaterials()
try: m.addGraph(PathTracerMaterials)
except NameError: None
