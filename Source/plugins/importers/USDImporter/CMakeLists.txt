if(NOT (FALCOR_ENABLE_USD AND FALCOR_HAS_NV_USD))
    return()
endif()

add_plugin(USDImporter)

target_sources(USDImporter PRIVATE
    ImporterContext.cpp
    ImporterContext.h
    USDImporter.cpp
    USDImporter.h
)

target_link_libraries(USDImporter PRIVATE nv-usd USDUtils)

target_copy_shaders(USDImporter plugins/importers/USDImporter)

target_source_group(USDImporter "Plugins/Importers")

validate_headers(USDImporter)
