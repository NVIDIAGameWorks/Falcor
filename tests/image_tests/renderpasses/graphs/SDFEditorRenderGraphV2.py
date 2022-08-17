from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph('DefaultRenderGraph')
    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary('PathTracer.dll')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('SDFEditor.dll')
    loadRenderPassLibrary('ToneMapper.dll')
    GBufferRT = createPass('GBufferRT')
    g.addPass(GBufferRT, 'GBufferRT')
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': AccumulatePrecision.Single})
    g.addPass(AccumulatePass, 'AccumulatePass')
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
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
