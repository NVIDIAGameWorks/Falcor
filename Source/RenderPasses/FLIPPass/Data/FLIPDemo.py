# Graphs
from falcor import *

def render_graph_FLIP_demo():
    g = RenderGraph('FLIPDemo')

    loadRenderPassLibrary('AccumulatePass.dll')
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("MinimalPathTracer.dll")
    loadRenderPassLibrary("FLIPPass.dll")
    loadRenderPassLibrary("ToneMapper.dll")

    FLIPPass = createPass("FLIPPass", {'enabled': True, 'useRealMonitorInfo': False, 'monitorDistanceMeters': 0.7, 'monitorWidthMeters': 0.5, 'monitorWidthPixels': 3840})
    g.addPass(FLIPPass, "FLIPPass")

    AccumulatePassA = createPass('AccumulatePass', {'enabled': True, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0})
    g.addPass(AccumulatePassA, 'AccumulatePassA')

    AccumulatePassB = createPass('AccumulatePass', {'enabled': True, 'autoReset': True, 'precisionMode': AccumulatePrecision.Single, 'subFrameCount': 0})
    g.addPass(AccumulatePassB, 'AccumulatePassB')

    MinimalPathTracerA = createPass("MinimalPathTracer", {'mMaxBounces': 3, 'mComputeDirect': False})
    g.addPass(MinimalPathTracerA, "MinimalPathTracerA")

    MinimalPathTracerB = createPass("MinimalPathTracer", {'mMaxBounces': 3, 'mComputeDirect': True})
    g.addPass(MinimalPathTracerB, "MinimalPathTracerB")

    GBufferRT = createPass("GBufferRT", {'samplePattern': SamplePattern.Stratified, 'sampleCount': 16})
    g.addPass(GBufferRT, "GBufferRT")

    ToneMappingPassA = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 3.0})
    g.addPass(ToneMappingPassA, "ToneMappingPassA")

    ToneMappingPassB = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMappingPassB, "ToneMappingPassB")

    g.addEdge("GBufferRT.posW", "MinimalPathTracerA.posW")
    g.addEdge("GBufferRT.normW", "MinimalPathTracerA.normalW")
    g.addEdge("GBufferRT.tangentW", "MinimalPathTracerA.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "MinimalPathTracerA.faceNormalW")
    g.addEdge("GBufferRT.viewW", "MinimalPathTracerA.viewW")
    g.addEdge("GBufferRT.diffuseOpacity", "MinimalPathTracerA.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "MinimalPathTracerA.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "MinimalPathTracerA.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "MinimalPathTracerA.mtlParams")

    g.addEdge("GBufferRT.posW", "MinimalPathTracerB.posW")
    g.addEdge("GBufferRT.normW", "MinimalPathTracerB.normalW")
    g.addEdge("GBufferRT.tangentW", "MinimalPathTracerB.tangentW")
    g.addEdge("GBufferRT.faceNormalW", "MinimalPathTracerB.faceNormalW")
    g.addEdge("GBufferRT.viewW", "MinimalPathTracerB.viewW")
    g.addEdge("GBufferRT.diffuseOpacity", "MinimalPathTracerB.mtlDiffOpacity")
    g.addEdge("GBufferRT.specRough", "MinimalPathTracerB.mtlSpecRough")
    g.addEdge("GBufferRT.emissive", "MinimalPathTracerB.mtlEmissive")
    g.addEdge("GBufferRT.matlExtra", "MinimalPathTracerB.mtlParams")

    g.addEdge("MinimalPathTracerA.color",  "AccumulatePassA.input")
    g.addEdge("MinimalPathTracerB.color",  "AccumulatePassB.input")

    # Tonemapping should be performed before LDR-FLIP
    g.addEdge("AccumulatePassA.output",  "ToneMappingPassA.src")
    g.addEdge("AccumulatePassB.output",  "ToneMappingPassB.src")

    g.addEdge("ToneMappingPassA.dst",  "FLIPPass.inputA")
    g.addEdge("ToneMappingPassB.dst",  "FLIPPass.inputB")

    g.markOutput('FLIPPass.output')
    return g

m.addGraph(render_graph_FLIP_demo())

# Scene
m.loadScene('Arcade/Arcade.pyscene')
