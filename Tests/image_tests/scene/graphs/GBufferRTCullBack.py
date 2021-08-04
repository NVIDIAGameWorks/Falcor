from falcor import *

def render_graph_GBufferRTCullBack():
    g = RenderGraph('GBufferRTCullBack')
    loadRenderPassLibrary('GBuffer.dll')
    GBufferRT = createPass('GBufferRT', {'samplePattern': SamplePattern.Center, 'forceCullMode': True, 'cull': CullMode.CullBack})
    g.addPass(GBufferRT, 'GBufferRT')
    g.markOutput('GBufferRT.faceNormalW')
    return g

GBufferRTCullBack = render_graph_GBufferRTCullBack()
try: m.addGraph(GBufferRTCullBack)
except NameError: None
