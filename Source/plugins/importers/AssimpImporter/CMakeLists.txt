add_plugin(AssimpImporter)

target_sources(AssimpImporter PRIVATE
    AssimpImporter.cpp
    AssimpImporter.h
)

target_link_libraries(AssimpImporter PRIVATE assimp)

target_source_group(AssimpImporter "Plugins/Importers")

validate_headers(AssimpImporter)
