from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_DefaultRenderGraph():
    g = RenderGraph('DefaultRenderGraph')
    g.create_pass('WireframePass', 'WireframePass', {})
    g.create_pass('ImageLoader', 'ImageLoader', {'outputSize': 'Default', 'filename': 'C:\\Users\\Admin\\Downloads\\test.jpg', 'mips': False, 'srgb': True, 'arrayIndex': 0, 'mipLevel': 0})
    g.add_edge('ImageLoader.dst', 'WireframePass.input')
    g.mark_output('WireframePass.output')
    return g

DefaultRenderGraph = render_graph_DefaultRenderGraph()
try: m.addGraph(DefaultRenderGraph)
except NameError: None
