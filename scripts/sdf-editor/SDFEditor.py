from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph('DefaultRenderGraph')
    GBufferRT = createPass('GBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'texLOD': 'Mip0', 'useDOF': False})
    g.addPass(GBufferRT, 'GBufferRT')
    AccumulatePass = createPass('AccumulatePass', {'enabled': True, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, 'AccumulatePass')
    ToneMapper = createPass('ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority', 'irayExposure': False})
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
