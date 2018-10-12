def render_graph_testDepthPass():
	testDepthPass = createRenderGraph()
	DepthPass = createRenderPass("DepthPass", {'depthFormat': Format.D32Float})
	testDepthPass.addPass(DepthPass, "DepthPass")
	BlitPass = createRenderPass("BlitPass", {'filter': Filter.Linear})
	testDepthPass.addPass(BlitPass, "BlitPass")
	testDepthPass.addEdge("DepthPass.depth", "BlitPass.src")
	testDepthPass.markOutput("BlitPass.dst")
	return testDepthPass

testDepthPass = render_graph_testDepthPass()