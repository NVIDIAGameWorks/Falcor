from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph('DefaultRenderGraph')
    GBufferRaster = createPass('GBufferRaster', {'samplePattern': 'Center', 'forceCullMode': True, 'cull': 'Back'})
    g.addPass(GBufferRaster, 'GBufferRaster')
    g.markOutput('GBufferRaster.faceNormalW')
    return g

DefaultRenderGraph = render_graph_DefaultRenderGraph()
try: m.addGraph(DefaultRenderGraph)
except NameError: None
