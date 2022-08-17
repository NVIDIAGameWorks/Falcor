add_renderpass(PassLibraryTemplate)

target_sources(PassLibraryTemplate PRIVATE
    PassLibraryTemplate.cpp
    PassLibraryTemplate.h
)

target_source_group(PassLibraryTemplate "RenderPasses")
