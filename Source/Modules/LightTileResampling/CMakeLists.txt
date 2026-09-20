add_library(LightTileResampling STATIC)

target_sources(LightTileResampling PRIVATE
    LightTileResampling.cpp
    LightTileResampling.h
    GenerateLightTiles.cs.slang
    Lights.slang
    LightSample.slang
    StreamingRIS.slang
    ReflectTypes.cs.slang
)

target_copy_shaders(LightTileResampling LightTileResampling)

target_source_group(LightTileResampling "Modules/")

target_link_libraries(LightTileResampling
    PUBLIC
        Falcor
)

target_include_directories(LightTileResampling
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(LightTileResampling
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
)

validate_headers(LightTileResampling)
