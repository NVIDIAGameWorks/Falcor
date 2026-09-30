# Graphs
from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_ReSTIR_FG():
    g = RenderGraph('ReSTIR_FG')
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': False, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Linear', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('VBufferRT', 'VBufferRT', {'outputSize': 'Default', 'samplePattern': 'Center', 'sampleCount': 16, 'useAlphaTest': True, 'adjustShadingNormals': True, 'forceCullMode': False, 'cull': 'Back', 'useTraceRayInline': False, 'useDOF': True})
    g.create_pass('ReSTIR_FG', 'ReSTIR_FG', {'PhotonBufferSizeGlobal': 400000, 'PhotonBufferSizeCaustic': 100000, 'AnalyticEmissiveRatio': 0.3499999940395355, 'PhotonBouncesGlobal': 10, 'PhotonBouncesCaustic': 10, 'PhotonRadiusGlobal': 0.019999999552965164, 'PhotonRadiusCaustic': 0.008500000461935997, 'EnableStochCollect': True, 'StochCollectK': 3, 'EnablePhotonCullingGlobal': False, 'EnablePhotonCullingCaustic': False, 'CullingRadius': 0.10000000149011612, 'CullingBits': 20, 'CausticCollectionMode': 3, 'CausticResamplingMode': 2, 'EnableDynamicDispatch': True, 'NumDispatchedPhotons': 1263104})
    g.add_edge('AccumulatePass.output', 'ToneMapper.src')
    g.add_edge('VBufferRT.mvec', 'ReSTIR_FG.mvec')
    g.add_edge('VBufferRT.vbuffer', 'ReSTIR_FG.vbuffer')
    g.add_edge('ReSTIR_FG.color', 'AccumulatePass.input')
    g.mark_output('ToneMapper.dst')
    g.mark_output('AccumulatePass.output')
    return g
m.addGraph(render_graph_ReSTIR_FG())

# Scene
m.loadScene('Bistro_ReSTIRFG/BistroInterior_Wine_EmissiveLight.pyscene')
m.scene.renderSettings = SceneRenderSettings(useEnvLight=True, useAnalyticLights=True, useEmissiveLights=True, useGridVolumes=True, diffuseAlbedoMultiplier=1)
m.scene.camera.position = float3(0.675186, 4.154675, -2.284865)
m.scene.camera.target = float3(1.602494, 3.902547, -2.561509)
m.scene.camera.up = float3(0.000859, 1.000000, -0.000256)
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

