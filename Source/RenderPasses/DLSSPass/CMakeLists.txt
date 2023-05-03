if(NOT (FALCOR_HAS_DLSS AND FALCOR_HAS_D3D12))
    return()
endif()

add_plugin(DLSSPass)

target_sources(DLSSPass PRIVATE
    DLSSPass.cpp
    DLSSPass.h
    NGXWrapper.cpp
    NGXWrapper.h
)

target_link_libraries(DLSSPass PRIVATE dlss)


target_source_group(DLSSPass "RenderPasses")
