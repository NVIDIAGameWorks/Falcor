add_renderpass(ImageLoader)

target_sources(ImageLoader PRIVATE
    ImageLoader.cpp
    ImageLoader.h
)

target_source_group(ImageLoader "RenderPasses")
