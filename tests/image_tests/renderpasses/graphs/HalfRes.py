from falcor import *

def render_graph_HalfRes():
    g = RenderGraph('HalfRes')

    GBuffer = createPass("GBufferRaster", {'outputSize': 'Half', 'samplePattern': 'Stratified'})
    g.addPass(GBuffer, "GBuffer")
    AccumulatePass = createPass("AccumulatePass", {'outputSize': 'Half', 'enabled': True})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass('ToneMapper', {'outputSize': 'Half'})
    g.addPass(ToneMapper, 'ToneMapper')
    PostFXPass = createPass("SimplePostFX", {'outputSize': 'Half', 'enabled': True, 'bloomAmount': 0.5})
    g.addPass(PostFXPass, "SimplePostFX")

    g.addEdge('GBuffer.normW', 'AccumulatePass.input')
    g.addEdge('AccumulatePass.output', 'ToneMapper.src')
    g.addEdge('ToneMapper.dst', 'SimplePostFX.src')

    g.markOutput('SimplePostFX.dst')
    g.markOutput('GBuffer.normW')
    g.markOutput('AccumulatePass.output')
    g.markOutput('ToneMapper.dst')

    return g

HalfRes = render_graph_HalfRes()
try: m.addGraph(HalfRes)
except NameError: None
