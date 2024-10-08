from falcor import *

def render_graph_PathTracerAdaptive():
    g = RenderGraph("PathTracerAdaptive")
    PathTracer = createPass("PathTracer", {'useSER': False})
    g.addPass(PathTracer, "PathTracer")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True})
    g.addPass(VBufferRT, "VBufferRT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    # Load a density map as an image in range [0,1] and scale it by 16.0
    DensityMap = createPass("ImageLoader", {'filename': 'test_images/density_map.png', 'mips': False, 'srgb': False})
    g.addPass(DensityMap, 'DensityMap')
    DensityScaler = createPass("Composite", {'scaleA': 16.0, 'outputFormat': 'Unknown'})
    g.addPass(DensityScaler, 'DensityScaler')
    g.addEdge('DensityMap.dst', 'DensityScaler.A')
    g.addEdge('DensityScaler.out', 'PathTracer.sampleCount')

    g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("PathTracer.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")

    # Final frame output
    g.markOutput("ToneMapper.dst")
    g.markOutput("ToneMapper.dst", TextureChannelFlags.Alpha)

    return g

PathTracerAdaptive = render_graph_PathTracerAdaptive()
try: m.addGraph(PathTracerAdaptive)
except NameError: None
