from falcor import *

def render_graph_GBufferRasterInspector():
    g = RenderGraph('GBufferRasterInspector')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('PixelInspectorPass.dll')
    GBufferRaster = createPass('GBufferRaster', {'samplePattern': SamplePattern.Center})
    g.addPass(GBufferRaster, 'GBufferRaster')
    PixelInspectorPass = createPass('PixelInspectorPass')
    g.addPass(PixelInspectorPass, 'PixelInspectorPass')
    g.addEdge('GBufferRaster.posW', 'PixelInspectorPass.posW')
    g.addEdge('GBufferRaster.normW', 'PixelInspectorPass.normW')
    g.addEdge('GBufferRaster.tangentW', 'PixelInspectorPass.tangentW')
    g.addEdge('GBufferRaster.faceNormalW', 'PixelInspectorPass.faceNormalW')
    g.addEdge('GBufferRaster.texC', 'PixelInspectorPass.texC')
    g.addEdge('GBufferRaster.texGrads', 'PixelInspectorPass.texGrads')
    g.addEdge('GBufferRaster.mtlData', 'PixelInspectorPass.mtlData')
    g.addEdge('GBufferRaster.vbuffer', 'PixelInspectorPass.vbuffer')
    g.addEdge('GBufferRaster', 'PixelInspectorPass')
    g.markOutput('GBufferRaster.faceNormalW')
    return g

GBufferRasterInspector = render_graph_GBufferRasterInspector()
try: m.addGraph(GBufferRasterInspector)
except NameError: None
