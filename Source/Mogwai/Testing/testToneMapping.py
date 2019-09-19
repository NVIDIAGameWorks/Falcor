def render_graph_testToneMapping():
	testToneMapping = RenderGraph("ToneMapper")
	ImageLoader = RenderPass("ImageLoader", {'fileName': '', 'mips': False, 'srgb': True, 'filename' : "StockImage.jpg"})
	testToneMapping.addPass(ImageLoader, "ImageLoader")
	ToneMapping = RenderPass("ToneMappingPass", {'operator': ToneMapOp.Aces})
	testToneMapping.addPass(ToneMapping, "ToneMapping")
	BlitPass = RenderPass("BlitPass", {'filter': SamplerFilter.Linear})
	testToneMapping.addPass(BlitPass, "BlitPass")
	testToneMapping.addEdge("ImageLoader.dst", "ToneMapping.src")
	testToneMapping.addEdge("ToneMapping.dst", "BlitPass.src")
	testToneMapping.markOutput("BlitPass.dst")
	return testToneMapping

testToneMapping = render_graph_testToneMapping()
try: m.addGraph(testToneMapping)
except NameError: None
