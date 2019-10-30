def render_graph_testForwardRendering():
	testForwardRendering = RenderGraph("ForwardRenderer")
	DepthPass = RenderPass("DepthPass", {'depthFormat': ResourceFormat.D32Float})
	testForwardRendering.addPass(DepthPass, "DepthPass")
	SkyBox = RenderPass("SkyBox")
	testForwardRendering.addPass(SkyBox, "SkyBox")
	ForwardLightingPass = RenderPass("ForwardLightingPass", {'sampleCount': 1, 'enableSuperSampling': False})
	testForwardRendering.addPass(ForwardLightingPass, "ForwardLightingPass")
	BlitPass = RenderPass("BlitPass", {'filter': SamplerFilter.Linear})
	testForwardRendering.addPass(BlitPass, "BlitPass")
	testForwardRendering.addEdge("ForwardLightingPass.color", "BlitPass.src")
	testForwardRendering.addEdge("DepthPass.depth", "ForwardLightingPass.depth")
	testForwardRendering.addEdge("DepthPass.depth", "SkyBox.depth")
	testForwardRendering.addEdge("SkyBox.target", "ForwardLightingPass.color")
	testForwardRendering.markOutput("BlitPass.dst")
	testForwardRendering.markOutput("ForwardLightingPass.motionVecs")
	return testForwardRendering

testForwardRendering = render_graph_testForwardRendering()
try: m.addGraph(testForwardRendering)
except NameError: None
