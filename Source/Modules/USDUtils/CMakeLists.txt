add_library(USDUtils STATIC)

target_sources(USDUtils PRIVATE
    PreviewSurfaceConverter/CreateSpecularTexture.cs.slang
    PreviewSurfaceConverter/CreateSpecularTransmissionTexture.cs.slang
    PreviewSurfaceConverter/PackBaseColorAlpha.cs.slang
    PreviewSurfaceConverter/PreviewSurfaceConverter.cpp
    PreviewSurfaceConverter/PreviewSurfaceConverter.h
    PreviewSurfaceConverter/SampleTexture.slang
    PreviewSurfaceConverter/StandardMaterialSpec.h

    Tessellator/IndexedVector.h
    Tessellator/Tessellation.cpp
    Tessellator/Tessellation.h

    USDHelpers.h
    USDScene1Utils.cpp
    USDScene1Utils.h
    USDUtils.h
    USDUtils.cpp
)

target_link_libraries(USDUtils
    PUBLIC
    nv-usd Falcor opensubdiv
)

target_include_directories(USDUtils
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

set_target_properties(USDUtils
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        LIBRARY_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
)

target_copy_shaders(USDUtils Modules/USDUtils)

target_source_group(USDUtils "Modules/")

validate_headers(USDUtils)
