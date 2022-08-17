add_renderpass(BSDFViewer)

target_sources(BSDFViewer PRIVATE
    BSDFViewer.cpp
    BSDFViewer.cs.slang
    BSDFViewer.h
    BSDFViewerParams.slang
)

target_copy_shaders(BSDFViewer RenderPasses/BSDFViewer)

target_source_group(BSDFViewer "RenderPasses")
