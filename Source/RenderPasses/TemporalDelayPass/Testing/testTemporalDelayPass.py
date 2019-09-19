def test_temporal_delay():
    imageLoader = RenderPass("ImageLoader", {'filename': 'smoke-puff.png', 'mips': False, 'srgb': True})
    depthPass = RenderPass("DepthPass")
    forwardLightingPass = RenderPass("ForwardLightingPass")
    temporalDelayPass = RenderPass("TemporalDelayPass", {"delay": 16})

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

temporal_delay_graph = test_temporal_delay()

m.addGraph(temporal_delay_graph)
