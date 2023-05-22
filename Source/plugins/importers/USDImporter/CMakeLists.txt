if(NOT (FALCOR_ENABLE_USD AND FALCOR_HAS_NV_USD))
    return()
endif()

add_plugin(USDImporter)

target_sources(USDImporter PRIVATE
    CreateSpecularTexture.cs.slang
    CreateSpecularTransmissionTexture.cs.slang
    ImporterContext.cpp
    ImporterContext.h
    IndexedVector.h
    PackBaseColorAlpha.cs.slang
    PreviewSurfaceConverter.cpp
    PreviewSurfaceConverter.h
    SampleTexture.slang
    StandardMaterialSpec.h
    Tessellation.cpp
    Tessellation.h
    USDHelpers.h
    USDImporter.cpp
    USDImporter.h
    Utils.h
)

target_link_libraries(USDImporter PRIVATE nv-usd opensubdiv)

target_copy_shaders(USDImporter plugins/importers/USDImporter)

target_source_group(USDImporter "Plugins/Importers")

validate_headers(USDImporter)
