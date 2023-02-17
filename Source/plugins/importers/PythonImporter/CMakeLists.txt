add_plugin(PythonImporter)

target_sources(PythonImporter PRIVATE
    PythonImporter.cpp
    PythonImporter.h
)

target_source_group(PythonImporter "Plugins/Importers")

validate_headers(PythonImporter)
