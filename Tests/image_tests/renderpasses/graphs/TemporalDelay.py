from falcor import *

def test_TemporalDelay():
    loadRenderPassLibrary("ImageLoader.dll")
    loadRenderPassLibrary("TemporalDelayPass.dll")
    imageLoader = createPass("ImageLoader", {'filename': 'smoke-puff.png', 'mips': False, 'srgb': True})
    depthPass = createPass("DepthPass")
    forwardLightingPass = createPass("ForwardLightingPass")
    temporalDelayPass = createPass("TemporalDelayPass", {"delay": 16})

    graph = RenderGraph("Temporal Delay Graph")
    graph.addPass(imageLoader, "ImageLoader")
    graph.addPass(depthPass, "DepthPass")
    graph.addPass(forwardLightingPass, "ForwardLightingPass")
    graph.addPass(temporalDelayPass, "TemporalDelayPass")

    graph.addEdge("ImageLoader.dst", "ForwardLightingPass.color")
    graph.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
    graph.addEdge("ForwardLightingPass.color", "TemporalDelayPass.src")
    graph.markOutput("TemporalDelayPass.maxDelay")

    return graph

TemporalDelay = test_TemporalDelay()
try: m.addGraph(TemporalDelay)
except NameError: None
