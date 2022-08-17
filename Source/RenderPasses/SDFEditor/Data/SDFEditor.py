from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph('DefaultRenderGraph')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('PathTracer.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('SDFEditor.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    GBufferRT = createPass('GBufferRT', {'outputSize': IOSize.Default, 'samplePattern': SamplePattern.Center, 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': CullMode.CullBack, 'texLOD': TexLODMode.Mip0, 'useDOF': False})
    g.addPass(GBufferRT, 'GBufferRT')
    AccumulatePass = createPass('AccumulatePass', {'enabled': True, 'outputSize': IOSize.Default, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0, 'maxAccumulatedFrames': 0})
    g.addPass(AccumulatePass, 'AccumulatePass')
    ToneMapper = createPass('ToneMapper', {'outputSize': IOSize.Default, 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': ToneMapOp.Aces, 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': ExposureMode.AperturePriority, 'irayExposure': False})
    g.addPass(ToneMapper, 'ToneMapper')
    SDFEditor = createPass('SDFEditor')
    g.addPass(SDFEditor, 'SDFEditor')
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1})
    g.addPass(PathTracer, 'PathTracer')
    g.addEdge('GBufferRT.vbuffer', 'PathTracer.vbuffer')
    g.addEdge('GBufferRT.viewW', 'PathTracer.viewW')
    g.addEdge('GBufferRT.mvecW', 'PathTracer.mvec')
    g.addEdge('GBufferRT.vbuffer', 'SDFEditor.vbuffer')
    g.addEdge('GBufferRT.linearZ', 'SDFEditor.linearZ')
    g.addEdge('PathTracer.color', 'AccumulatePass.input')
    g.addEdge('AccumulatePass.output', 'ToneMapper.src')
    g.addEdge('ToneMapper.dst', 'SDFEditor.inputColor')
    g.markOutput('SDFEditor.output')
    return g

DefaultRenderGraph = render_graph_DefaultRenderGraph()
try: m.addGraph(DefaultRenderGraph)
except NameError: None
