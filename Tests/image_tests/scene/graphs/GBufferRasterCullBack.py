from falcor import *

def render_graph_GBufferRasterCullBack():
    g = RenderGraph('GBufferRasterCullBack')
    loadRenderPassLibrary('GBuffer.dll')
    GBufferRaster = createPass('GBufferRaster', {'samplePattern': SamplePattern.Center, 'forceCullMode': True, 'cull': CullMode.CullBack})
    g.addPass(GBufferRaster, 'GBufferRaster')
    g.markOutput('GBufferRaster.faceNormalW')
    return g

GBufferRasterCullBack = render_graph_GBufferRasterCullBack()
try: m.addGraph(GBufferRasterCullBack)
except NameError: None
