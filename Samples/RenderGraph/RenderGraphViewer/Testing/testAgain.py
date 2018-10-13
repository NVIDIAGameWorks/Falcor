def render_graph_DefaultRenderGraph():
	DefaultRenderGraph = createRenderGraph()
	ImageLoader = createRenderPass("ImageLoader", {'fileName': 'C:\\Users\\moakes\\Pictures\\test.jpg', 'mips': True, 'srgb': True})
	DefaultRenderGraph.addPass(ImageLoader, "ImageLoader")
	BlitPass = createRenderPass("BlitPass", {'filter': Filter.Linear})
	DefaultRenderGraph.addPass(BlitPass, "BlitPass")
	DefaultRenderGraph.addEdge("ImageLoader.dst", "BlitPass.src")
	DefaultRenderGraph.markOutput("BlitPass.dst")
	return DefaultRenderGraph

DefaultRenderGraph = render_graph_DefaultRenderGraph()