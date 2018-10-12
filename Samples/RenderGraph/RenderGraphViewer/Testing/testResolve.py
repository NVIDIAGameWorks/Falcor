def render_graph_testResolve():
	testResolve = createRenderGraph()
	ImageLoader = createRenderPass("ImageLoader", {'fileName': '', 'mips': True, 'srgb': False})
	testResolve.addPass(ImageLoader, "ImageLoader")
	ResolvePass = createRenderPass("ResolvePass")
	testResolve.addPass(ResolvePass, "ResolvePass")
	BlitPass = createRenderPass("BlitPass", {'filter': Filter.Linear})
	testResolve.addPass(BlitPass, "BlitPass")
	testResolve.addEdge("ImageLoader.dst", "ResolvePass.src")
	testResolve.addEdge("ResolvePass.dst", "BlitPass.src")
	testResolve.markOutput("BlitPass.dst")
	return testResolve

testResolve = render_graph_testResolve()