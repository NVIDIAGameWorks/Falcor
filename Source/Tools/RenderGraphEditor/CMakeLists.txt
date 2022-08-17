add_falcor_executable(RenderGraphEditor)

set_target_properties(RenderGraphEditor PROPERTIES WIN32_EXECUTABLE TRUE)

target_sources(RenderGraphEditor PRIVATE
    RenderGraphEditor.cpp
    RenderGraphEditor.h
)

target_copy_data_folder(RenderGraphEditor)

target_source_group(RenderGraphEditor "Tools")
