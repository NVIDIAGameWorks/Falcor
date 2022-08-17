from falcor import *

def render_graph_SVGF():
    g = RenderGraph("SVGF")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("PathTracer.dll")
    loadRenderPassLibrary("SVGFPass.dll")

    SVGFPass = createPass("SVGFPass", {'Enabled': True, 'Iterations': 4, 'FeedbackTap': 1, 'VarianceEpsilon': 9.999999747378752e-05, 'PhiColor': 10.0, 'PhiNormal': 128.0, 'Alpha': 0.05000000074505806, 'MomentsAlpha': 0.20000000298023224})
    g.addPass(SVGFPass, "SVGFPass")
    GBufferRaster = createPass("GBufferRaster", {'cull': CullMode.CullBack})
    g.addPass(GBufferRaster, "GBufferRaster")
    PathTracer = createPass("PathTracer")
    g.addPass(PathTracer, "PathTracer")

    g.addEdge("PathTracer.color", "SVGFPass.Color")
    g.addEdge("PathTracer.albedo", "SVGFPass.Albedo")
    g.addEdge("GBufferRaster.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRaster.emissive", "SVGFPass.Emission")
    g.addEdge("GBufferRaster.posW", "SVGFPass.WorldPosition")
    g.addEdge("GBufferRaster.normW", "SVGFPass.WorldNormal")
    g.addEdge("GBufferRaster.pnFwidth", "SVGFPass.PositionNormalFwidth")
    g.addEdge("GBufferRaster.linearZ", "SVGFPass.LinearZ")
    g.addEdge("GBufferRaster.mvec", "SVGFPass.MotionVec")

    g.markOutput("SVGFPass.Filtered image")
    g.markOutput("PathTracer.color")
    g.markOutput("PathTracer.albedo")
    g.markOutput("GBufferRaster.posW")
    g.markOutput("GBufferRaster.normW")
    g.markOutput("GBufferRaster.emissive")
    g.markOutput("GBufferRaster.pnFwidth")
    g.markOutput("GBufferRaster.linearZ")
    g.markOutput("GBufferRaster.mvec")
    return g

SVGF = render_graph_SVGF()
try: m.addGraph(SVGF)
except NameError: None
