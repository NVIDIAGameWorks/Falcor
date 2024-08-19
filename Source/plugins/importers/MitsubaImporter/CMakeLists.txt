add_plugin(MitsubaImporter)

target_sources(MitsubaImporter PRIVATE
    MitsubaImporter.cpp
    MitsubaImporter.h
    Parser.h
    Resolver.h
    Tables.h
)

set(DEP_DIR ../../../../external)

target_include_directories(MitsubaImporter PRIVATE ${DEP_DIR}/packman/deps/include)
target_link_directories(MitsubaImporter PRIVATE ${DEP_DIR}/packman/deps/lib)
target_link_libraries(MitsubaImporter PRIVATE pugixml)

target_copy_shaders(MitsubaImporter plugins/importers/MitsubaImporter)

target_source_group(MitsubaImporter "Plugins/Importers")

validate_headers(MitsubaImporter)
