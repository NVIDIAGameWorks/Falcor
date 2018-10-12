def render_graph_testSkybox():
	testSkybox = createRenderGraph()
	DepthPass = createRenderPass("DepthPass", {'depthFormat': Format.D32Float})
	testSkybox.addPass(DepthPass, "DepthPass")
	SkyBox = createRenderPass("SkyBox")
	testSkybox.addPass(SkyBox, "SkyBox")
	BlitPass = createRenderPass("BlitPass", {'filter': Filter.Linear})
	testSkybox.addPass(BlitPass, "BlitPass")
	testSkybox.addEdge("DepthPass.depth", "SkyBox.depth")
	testSkybox.addEdge("SkyBox.target", "BlitPass.src")
	testSkybox.markOutput("BlitPass.dst")
	return testSkybox

testSkybox = render_graph_testSkybox()