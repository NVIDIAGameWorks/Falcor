from falcor import *

def render_graph_SceneDebugger():
    g = RenderGraph('SceneDebugger')
    SceneDebugger = createPass('SceneDebugger')
    g.addPass(SceneDebugger, 'SceneDebugger')
    g.markOutput('SceneDebugger.output')
    return g

SceneDebugger = render_graph_SceneDebugger()
try: m.addGraph(SceneDebugger)
except NameError: None
