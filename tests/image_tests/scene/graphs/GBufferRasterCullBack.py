from falcor import *

def render_graph_GBufferRasterCullBack():
    g = RenderGraph('GBufferRasterCullBack')
    GBufferRaster = createPass('GBufferRaster', {'samplePattern': 'Center', 'forceCullMode': True, 'cull': 'Back'})
    g.addPass(GBufferRaster, 'GBufferRaster')
    g.markOutput('GBufferRaster.faceNormalW')
    return g

GBufferRasterCullBack = render_graph_GBufferRasterCullBack()
try: m.addGraph(GBufferRasterCullBack)
except NameError: None
