if(NOT FALCOR_HAS_DLSS)
    return()
endif()

add_renderpass(DLSSPass)

target_sources(DLSSPass PRIVATE
    DLSSPass.cpp
    DLSSPass.h
    NGXWrapper.cpp
    NGXWrapper.h
)

target_link_libraries(DLSSPass PRIVATE ngx)

target_source_group(DLSSPass "RenderPasses")
