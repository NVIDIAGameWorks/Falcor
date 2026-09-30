# Graphs
from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_ReSTIR_FG_NRD():
    g = RenderGraph('ReSTIR_FG_NRD')
    g.create_pass('ModulateIllumination', 'ModulateIllumination', {'useEmission': True, 'useDiffuseReflectance': True, 'useDiffuseRadiance': True, 'useSpecularReflectance': True, 'useSpecularRadiance': True, 'useDeltaReflectionEmission': False, 'useDeltaReflectionReflectance': False, 'useDeltaReflectionRadiance': False, 'useDeltaTransmissionEmission': False, 'useDeltaTransmissionReflectance': False, 'useDeltaTransmissionRadiance': False, 'useResidualRadiance': True, 'outputSize': 'Default', 'useDebug': True, 'inputInYCoCg': False})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': False, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('DLSSPass', 'DLSSPass', {'enabled': True, 'outputSize': 'Default', 'profile': 'Balanced', 'motionVectorScale': 'Relative', 'isHDR': True, 'useJitteredMV': False, 'sharpness': 0.0, 'exposure': 0.0})
    g.create_pass('ReSTIR_FG', 'ReSTIR_FG', {'PhotonBufferSizeGlobal': 800000, 'PhotonBufferSizeCaustic': 400000, 'AnalyticEmissiveRatio': 0.3499999940395355, 'PhotonBouncesGlobal': 10, 'PhotonBouncesCaustic': 10, 'PhotonRadiusGlobal': 0.019999999552965164, 'PhotonRadiusCaustic': 0.007000000216066837, 'EnableStochCollect': True, 'StochCollectK': 3, 'EnablePhotonCullingGlobal': True, 'EnablePhotonCullingCaustic': False, 'CullingRadius': 0.10000000149011612, 'CullingBits': 20, 'CausticCollectionMode': 3, 'CausticResamplingMode': 2, 'EnableDynamicDispatch': True, 'NumDispatchedPhotons': 2000000})
    g.create_pass('NRD', 'NRD_Normal', {'enabled': True, 'method': 'RelaxDiffuseSpecular', 'outputSize': 'Default', 'worldSpaceMotion': True, 'disocclusionThreshold': 2.0, 'maxIntensity': 250.0, 'RelaxAntilagAccelerationAmount': 0.30000001192092896, 'RelaxAntilagSpatialSigmaScale': 4.0, 'RelaxAntilagTemporalSigmaScale': 0.20000000298023224, 'RelaxAntilagResetAmount': 0.699999988079071, 'RelaxDiffusePrepassBlurRadius': 16.0, 'RelaxSpecularPrepassBlurRadius': 16.0, 'RelaxDiffuseMaxAccumulatedFrameNum': 40, 'RelaxSpecularMaxAccumulatedFrameNum': 40, 'RelaxEnableAntiFirefly': True})
    g.create_pass('GBufferRT', 'GBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'texLOD': 'Mip0', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Linear', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.add_edge('ReSTIR_FG.residualRadiance', 'ModulateIllumination.residualRadiance')
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.add_edge('NRD.filteredSpecularRadianceHitDist', 'ModulateIllumination.specularRadiance')
    g.add_edge('ReSTIR_FG.emission', 'ModulateIllumination.emission')
    g.add_edge('ReSTIR_FG.diffuseReflectance', 'ModulateIllumination.diffuseReflectance')
    g.add_edge('ReSTIR_FG.specularReflectance', 'ModulateIllumination.specularReflectance')
    g.add_edge('NRD.filteredDiffuseRadianceHitDist', 'ModulateIllumination.diffuseRadiance')
    g.add_edge('ReSTIR_FG.diffuseRadiance', 'NRD.diffuseRadianceHitDist')
    g.add_edge('ReSTIR_FG.specularRadiance', 'NRD.specularRadianceHitDist')
    g.add_edge('GBufferRT.linearZ', 'NRD.viewZ')
    g.add_edge('GBufferRT.normWRoughnessMaterialID', 'NRD.normWRoughnessMaterialID')
    g.add_edge('ModulateIllumination.output', 'DLSSPass.color')
    g.add_edge('DLSSPass.output', 'AccumulatePass.input')
    g.add_edge('GBufferRT.depth', 'DLSSPass.depth')
    g.add_edge('GBufferRT.mvecW', 'NRD.mvec')
    g.add_edge('GBufferRT.mvec', 'DLSSPass.mvec')
    g.add_edge('GBufferRT.vbuffer', 'ReSTIR_FG.vbuffer')
    g.add_edge('GBufferRT.mvec', 'ReSTIR_FG.mvec')
    g.add_edge('NRD.outValidation', 'ModulateIllumination.debugLayer')
    g.mark_output('ToneMapper.dst')
    g.mark_output('AccumulatePass.output')
    return g
m.addGraph(render_graph_ReSTIR_FG_NRD())

# Scene
m.loadScene('VeachAjar/VeachAjar.pyscene')
m.scene.cameraSpeed = 1.0

# Window Configuration
m.resizeFrameBuffer(1920, 1080)
m.ui = True

# Clock Settings
m.clock.time = 0
m.clock.framerate = 0
# If framerate is not zero, you can use the frame property to set the start frame
# m.clock.frame = 0

# Frame Capture
m.frameCapture.outputDir = '.'
m.frameCapture.baseFilename = 'Mogwai'

