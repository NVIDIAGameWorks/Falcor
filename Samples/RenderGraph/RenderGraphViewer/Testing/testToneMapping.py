def render_graph_testToneMapping():
	testToneMapping = createRenderGraph()
	ImageLoader = createRenderPass("ImageLoader", {'fileName': '', 'mips': True, 'srgb': True})
	testToneMapping.addPass(ImageLoader, "ImageLoader")
	ToneMapping = createRenderPass("ToneMapping", {'operator': ToneMapOp.Aces})
	testToneMapping.addPass(ToneMapping, "ToneMapping")
	BlitPass = createRenderPass("BlitPass", {'filter': Filter.Linear})
	testToneMapping.addPass(BlitPass, "BlitPass")
	testToneMapping.addEdge("ImageLoader.dst", "ToneMapping.src")
	testToneMapping.addEdge("ToneMapping.dst", "BlitPass.src")
	testToneMapping.markOutput("BlitPass.dst")
	return testToneMapping

testToneMapping = render_graph_testToneMapping()