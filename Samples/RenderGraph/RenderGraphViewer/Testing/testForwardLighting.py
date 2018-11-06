def render_graph_testForwardLighting():
	testForwardLighting = createRenderGraph()
	DepthPass = createRenderPass("DepthPass", {'depthFormat': Format.D32Float})
	testForwardLighting.addPass(DepthPass, "DepthPass")
	SkyBox = createRenderPass("SkyBox")
	testForwardLighting.addPass(SkyBox, "SkyBox")
	ForwardLightingPass = createRenderPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
	testForwardLighting.addPass(ForwardLightingPass, "ForwardLightingPass")
	BlitPass = createRenderPass("BlitPass", {'filter': Filter.Linear})
	testForwardLighting.addPass(BlitPass, "BlitPass")
	testForwardLighting.addEdge("ForwardLightingPass.color", "BlitPass.src")
	testForwardLighting.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
	testForwardLighting.addEdge("DepthPass.depth", "SkyBox.depth")
	testForwardLighting.addEdge("SkyBox.target", "ForwardLightingPass.color")
	testForwardLighting.markOutput("BlitPass.dst")
	return testForwardLighting

testForwardLighting = render_graph_testForwardLighting()