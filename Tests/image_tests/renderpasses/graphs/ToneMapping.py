from falcor import *

def render_graph_ToneMapping():
    loadRenderPassLibrary("ImageLoader.dll")
    loadRenderPassLibrary("ToneMapper.dll")
    loadRenderPassLibrary("BlitPass.dll")
    testToneMapping = RenderGraph("ToneMapper")
    ImageLoader = RenderPass("ImageLoader", {'filename' : "LightProbes/hallstatt4_hd.hdr", 'mips': False, 'srgb': True})
    testToneMapping.addPass(ImageLoader, "ImageLoader")
    ToneMapping = RenderPass("ToneMapper")
    testToneMapping.addPass(ToneMapping, "ToneMapping")
    BlitPass = RenderPass("BlitPass", {'filter': SamplerFilter.Linear})
    testToneMapping.addPass(BlitPass, "BlitPass")
    testToneMapping.addEdge("ImageLoader.dst", "ToneMapping.src")
    testToneMapping.addEdge("ToneMapping.dst", "BlitPass.src")
    testToneMapping.markOutput("BlitPass.dst")
    return testToneMapping

ToneMapping = render_graph_ToneMapping()
try: m.addGraph(ToneMapping)
except NameError: None
