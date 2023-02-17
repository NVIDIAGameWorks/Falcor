add_falcor_executable(RenderGraphEditor)

set_target_properties(RenderGraphEditor PROPERTIES WIN32_EXECUTABLE TRUE)

target_sources(RenderGraphEditor PRIVATE
    RenderGraphEditor.cpp
    RenderGraphEditor.h
)

target_link_libraries(RenderGraphEditor PRIVATE args)

target_source_group(RenderGraphEditor "Tools")
