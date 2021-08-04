from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph('DefaultRenderGraph')
    loadRenderPassLibrary('GBuffer.dll')
    loadRenderPassLibrary('PixelInspectorPass.dll')
    GBufferRaster = createPass('GBufferRaster', {'samplePattern': SamplePattern.Center})
    g.addPass(GBufferRaster, 'GBufferRaster')
    PixelInspectorPass = createPass('PixelInspectorPass')
    g.addPass(PixelInspectorPass, 'PixelInspectorPass')
    g.addEdge('GBufferRaster.posW', 'PixelInspectorPass.posW')
    g.addEdge('GBufferRaster.normW', 'PixelInspectorPass.normW')
    g.addEdge('GBufferRaster.tangentW', 'PixelInspectorPass.tangentW')
    g.addEdge('GBufferRaster.texC', 'PixelInspectorPass.texC')
    g.addEdge('GBufferRaster.diffuseOpacity', 'PixelInspectorPass.diffuseOpacity')
    g.addEdge('GBufferRaster.specRough', 'PixelInspectorPass.specRough')
    g.addEdge('GBufferRaster.emissive', 'PixelInspectorPass.emissive')
    g.addEdge('GBufferRaster.matlExtra', 'PixelInspectorPass.matlExtra')
    g.addEdge('GBufferRaster.faceNormalW', 'PixelInspectorPass.faceNormalW')
    g.addEdge('GBufferRaster.vbuffer', 'PixelInspectorPass.vbuffer')
    g.addEdge('GBufferRaster', 'PixelInspectorPass')
    g.markOutput('GBufferRaster.faceNormalW')
    return g

DefaultRenderGraph = render_graph_DefaultRenderGraph()
try: m.addGraph(DefaultRenderGraph)
except NameError: None
