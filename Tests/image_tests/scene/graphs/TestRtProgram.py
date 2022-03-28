from falcor import *

def render_graph_TestRtProgram():
    g = RenderGraph('TestRtProgramGraph')
    loadRenderPassLibrary('TestPasses.dll')
    TestRtProgram = createPass('TestRtProgram', {'mode': 0})
    g.addPass(TestRtProgram, 'TestRtProgram')
    g.markOutput('TestRtProgram.output')
    return g

TestRtProgramGraph = render_graph_TestRtProgram()
try: m.addGraph(TestRtProgramGraph)
except NameError: None
