from falcor import *

def test_ColorMapPass():
    imageLoader = createPass("ImageLoader", {'filename' : "test_scenes/envmaps/hallstatt4_hd.hdr", 'mips': False, 'srgb': False, 'outputFormat': 'RGBA32Float'})
    colorMap = createPass("ColorMapPass")

    graph = RenderGraph("Color Map")
    graph.addPass(imageLoader, "ImageLoader")
    graph.addPass(colorMap, "ColorMap")

    graph.addEdge("ImageLoader.dst", "ColorMap.input")
    graph.markOutput("ColorMap.output")

    return graph

ColorMapPass = test_ColorMapPass()
try: m.addGraph(ColorMapPass)
except NameError: None
