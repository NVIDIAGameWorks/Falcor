from falcor import *

def render_graph_PathTracerNRD():
    g = RenderGraph("PathTracerNRD")

    loadRenderPassLibrary("AccumulatePass.dll")
    loadRenderPassLibrary("DLSSPass.dll")
    loadRenderPassLibrary("GBuffer.dll")
    loadRenderPassLibrary("ModulateIllumination.dll")
    loadRenderPassLibrary("NRDPass.dll")
    loadRenderPassLibrary("PathTracer.dll")
    loadRenderPassLibrary("ToneMapper.dll")

    GBufferRT = createPass("GBufferRT", {'samplePattern': SamplePattern.Halton, 'sampleCount': 32, 'useAlphaTest': True})
    g.addPass(GBufferRT, "GBufferRT")
    PathTracer = createPass("PathTracer", {'samplesPerPixel': 1, 'maxSurfaceBounces': 10, 'useRussianRoulette': True})
    g.addPass(PathTracer, "PathTracer")

    # Reference path passes
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': AccumulatePrecision.Single})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapperReference = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapperReference, "ToneMapperReference")

    # NRD path passes
    NRD = createPass("NRD", {'enableRoughnessBasedSpecularAccumulation': False})
    g.addPass(NRD, "NRD")
    NRDResidual = createPass("NRD", {'maxIntensity': 50, 'enableRoughnessBasedSpecularAccumulation': False})
    g.addPass(NRDResidual, "NRDResidual")
    ModulateIllumination = createPass("ModulateIllumination")
    g.addPass(ModulateIllumination, "ModulateIllumination")
    DLSS = createPass("DLSSPass", {'enabled': True, 'profile': DLSSProfile.Balanced, 'motionVectorScale': DLSSMotionVectorScale.Relative, 'isHDR': True, 'sharpness': 0.0, 'exposure': 0.0})
    g.addPass(DLSS, "DLSS")
    ToneMapperNRD = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapperNRD, "ToneMapperNRD")

    g.addEdge("GBufferRT.vbuffer",                          "PathTracer.vbuffer")
    g.addEdge("GBufferRT.viewW",                            "PathTracer.viewW")

    # Reference path graph
    g.addEdge("PathTracer.color",                           "AccumulatePass.input")
    g.addEdge("AccumulatePass.output",                      "ToneMapperReference.src")

    # NRD path graph
    g.addEdge("PathTracer.nrdDiffuseRadianceHitDist",       "NRD.diffuseRadianceHitDist")
    g.addEdge("PathTracer.nrdSpecularRadianceHitDist",      "NRD.specularRadianceHitDist")
    g.addEdge("GBufferRT.mvecW",                            "NRD.mvec")
    g.addEdge("GBufferRT.normWRoughnessMaterialID",         "NRD.normWRoughnessMaterialID")
    g.addEdge("GBufferRT.linearZ",                          "NRD.viewZ")

    g.addEdge("PathTracer.nrdResidualRadianceHitDist",      "NRDResidual.diffuseRadianceHitDist")
    g.addEdge("PathTracer.nrdResidualRadianceHitDist",      "NRDResidual.specularRadianceHitDist")
    g.addEdge("GBufferRT.mvecW",                            "NRDResidual.mvec")
    g.addEdge("GBufferRT.normWRoughnessMaterialID",         "NRDResidual.normWRoughnessMaterialID")
    g.addEdge("GBufferRT.linearZ",                          "NRDResidual.viewZ")

    g.addEdge("PathTracer.nrdEmission",                     "ModulateIllumination.emission")
    g.addEdge("PathTracer.nrdDiffuseReflectance",           "ModulateIllumination.diffuseReflectance")
    g.addEdge("NRD.filteredDiffuseRadianceHitDist",         "ModulateIllumination.diffuseRadiance")
    g.addEdge("PathTracer.nrdSpecularReflectance",          "ModulateIllumination.specularReflectance")
    g.addEdge("NRD.filteredSpecularRadianceHitDist",        "ModulateIllumination.specularRadiance")
    g.addEdge("NRDResidual.filteredDiffuseRadianceHitDist", "ModulateIllumination.residualRadiance")

    g.addEdge("GBufferRT.mvec",                             "DLSS.mvec")
    g.addEdge("GBufferRT.linearZ",                          "DLSS.depth")
    g.addEdge("ModulateIllumination.output",                "DLSS.color")

    g.addEdge("DLSS.output",                                "ToneMapperNRD.src")

    # Outputs
    g.markOutput("ToneMapperNRD.dst")
    g.markOutput("ToneMapperReference.dst")

    return g

PathTracerNRD = render_graph_PathTracerNRD()
try: m.addGraph(PathTracerNRD)
except NameError: None
