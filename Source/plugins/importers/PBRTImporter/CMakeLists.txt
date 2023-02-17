add_plugin(PBRTImporter)

target_sources(PBRTImporter PRIVATE
    Builder.cpp
    Builder.h
    EnvMapConverter.cs.slang
    EnvMapConverter.h
    Helpers.h
    LoopSubdivide.cpp
    LoopSubdivide.h
    Parameters.cpp
    Parameters.h
    Parser.cpp
    Parser.h
    PBRTImporter.cpp
    PBRTImporter.h
    Types.h
)

target_copy_shaders(PBRTImporter plugins/importers/PBRTImporter)

target_source_group(PBRTImporter "Plugins/Importers")

validate_headers(PBRTImporter)
