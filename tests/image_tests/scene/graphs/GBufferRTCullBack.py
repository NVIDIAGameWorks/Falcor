from falcor import *

def render_graph_GBufferRTCullBack():
    g = RenderGraph('GBufferRTCullBack')
    GBufferRT = createPass('GBufferRT', {'samplePattern': 'Center', 'forceCullMode': True, 'cull': 'Back'})
    g.addPass(GBufferRT, 'GBufferRT')
    g.markOutput('GBufferRT.faceNormalW')
    return g

GBufferRTCullBack = render_graph_GBufferRTCullBack()
try: m.addGraph(GBufferRTCullBack)
except NameError: None
