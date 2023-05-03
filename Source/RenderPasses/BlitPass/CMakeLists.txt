add_plugin(BlitPass)

target_sources(BlitPass PRIVATE
    BlitPass.cpp
    BlitPass.h
)

target_source_group(BlitPass "RenderPasses")
