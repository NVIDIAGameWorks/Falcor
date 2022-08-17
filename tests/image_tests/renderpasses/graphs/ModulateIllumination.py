from falcor import *

def render_graph_ModulateIllumination():
    loadRenderPassLibrary("ModulateIllumination.dll")
    g = RenderGraph("ModulateIllumination")
    ModulateIllumination = createPass("ModulateIllumination")
    g.addPass(ModulateIllumination, "ModulateIllumination")
    ImageLoaderA = createPass("ImageLoader", {'filename': 'Cubemaps\\Sorsele3\\posz.jpg', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoaderA, "ImageLoaderA")
    ImageLoaderB = createPass("ImageLoader", {'filename': 'smoke-puff.png', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoaderB, "ImageLoaderB")
    ImageLoaderC = createPass("ImageLoader", {'filename': 'Cubemaps\\Sorsele3\\posy.jpg', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.addPass(ImageLoaderC, "ImageLoaderC")
    g.addEdge("ImageLoaderA.dst", "ModulateIllumination.diffuseReflectance")
    g.addEdge("ImageLoaderB.dst", "ModulateIllumination.diffuseRadiance")
    g.addEdge("ImageLoaderB.dst", "ModulateIllumination.specularRadiance")
    g.markOutput("ModulateIllumination.output")
    return g

ModulateIllumination = render_graph_ModulateIllumination()
try: m.addGraph(ModulateIllumination)
except NameError: None
