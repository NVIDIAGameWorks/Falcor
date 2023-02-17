add_falcor_executable(ImageCompare)

target_sources(ImageCompare PRIVATE
    ImageCompare.cpp
)

target_link_libraries(ImageCompare PRIVATE args FreeImage)

target_source_group(ImageCompare "Tools")
