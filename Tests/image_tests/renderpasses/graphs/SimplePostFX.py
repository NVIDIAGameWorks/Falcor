from falcor import *

def render_graph_SimplePostFX():
    loadRenderPassLibrary("ImageLoader.dll")
    loadRenderPassLibrary("BlitPass.dll")
    loadRenderPassLibrary("SimplePostFX.dll")
    testSimplePostFX = RenderGraph("SimplePostFX")
    ImageLoader = createPass("ImageLoader", {'filename' : "LightProbes/20060807_wells6_hd.hdr", 'mips': False, 'srgb': False})
    testSimplePostFX.addPass(ImageLoader, "ImageLoader")
    PostFXPass = createPass("SimplePostFX",
        {'bloomAmount': 0.5,
        'starAmount': 0.3,
        'vignetteAmount': 0.3,
        'chromaticAberrationAmount': 0.5,
        'barrelDistortAmount': 0.1 ,
        'saturationCurve': float3(0.5, 0.75, 1.),
        'colorOffset': float3(0.4, 0.4, 0.5),
        'colorScale': float3(0.4,0.3,0.2),
        'colorPower': float3(0.4,0.5,0.6),
        'colorOffsetScalar': 0.1,
        'colorScaleScalar': 1.0,
        'colorPowerScalar': 0.13,
        'enabled': True,
        'wipe': 0.33})
    testSimplePostFX.addPass(PostFXPass, "SimplePostFX")
    BlitPass = createPass("BlitPass", {'filter': SamplerFilter.Linear})
    testSimplePostFX.addPass(BlitPass, "BlitPass")
    testSimplePostFX.addEdge("ImageLoader.dst", "SimplePostFX.src")
    testSimplePostFX.addEdge("SimplePostFX.dst", "BlitPass.src")
    testSimplePostFX.markOutput("BlitPass.dst")
    return testSimplePostFX

SimplePostFX = render_graph_SimplePostFX()
try: m.addGraph(SimplePostFX)
except NameError: None
