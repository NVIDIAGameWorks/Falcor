if(NOT (FALCOR_HAS_NRD AND FALCOR_HAS_D3D12))
    return()
endif()

add_renderpass(NRDPass)

target_sources(NRDPass PRIVATE
    NRDPass.cpp
    NRDPass.h
    PackRadiance.cs.slang
)

target_link_libraries(NRDPass PRIVATE nrd)

target_copy_shaders(NRDPass RenderPasses/NRDPass)

target_source_group(NRDPass "RenderPasses")
