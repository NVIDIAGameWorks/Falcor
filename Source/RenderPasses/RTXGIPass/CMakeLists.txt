if(NOT FALCOR_HAS_D3D12)
    return()
endif()

add_renderpass(RTXGIPass)

target_sources(RTXGIPass PRIVATE
    ComputeIndirect.cs.slang
    LoadShadingData.slang
    RTXGIPass.cpp
    RTXGIPass.h
    VisualizeBackground.cs.slang
    VisualizeDirect.rt.slang
    VisualizeProbes.3d.slang
    VisualizerDefines.slang
)

target_copy_shaders(RTXGIPass RenderPasses/RTXGIPass)
target_copy_data_folder(RTXGIPass RenderPasses/RTXGIPass)

target_source_group(RTXGIPass "RenderPasses")
