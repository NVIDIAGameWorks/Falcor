def render_graph_TestCSM():
	TestCSM = createRenderGraph()
	DepthPass = createRenderPass("DepthPass", {'depthFormat': Format.D32Float})
	TestCSM.addPass(DepthPass, "DepthPass")
	CascadedShadowMaps = createRenderPass("CascadedShadowMaps")
	TestCSM.addPass(CascadedShadowMaps, "CascadedShadowMaps")
	BlitPass = createRenderPass("BlitPass", {'filter': Filter.Linear})
	TestCSM.addPass(BlitPass, "BlitPass")
	TestCSM.addEdge("DepthPass.depth", "CascadedShadowMaps.depth")
	TestCSM.addEdge("CascadedShadowMaps.visibility", "BlitPass.src")
	TestCSM.markOutput("BlitPass.dst")
	return TestCSM

TestCSM = render_graph_TestCSM()