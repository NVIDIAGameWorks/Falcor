/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "PythonImporter.h"
#include "Scene/Importer.h"
#include "Utils/Scripting/Scripting.h"
#include <filesystem>
#include <regex>

namespace Falcor
{
    namespace
    {
        /** Parse the legacy header on the first line of the script with the syntax:
            # filename.extension
        */
        static std::optional<std::string> parseLegacyHeader(const std::string& script)
        {
            if (size_t endOfFirstLine = script.find_first_of("\n\r"); endOfFirstLine != std::string::npos)
            {
                const std::regex headerRegex(R"""(#\s+([\w-]+\.[\w]{1,10}))""");

                std::smatch match;
                if (std::regex_match(script.begin(), script.begin() + endOfFirstLine, match, headerRegex)) {
                    if (match.size() > 1) return match[1].str();
                }
            }

            return {};
        }

        static std::set<std::filesystem::path> sImportPaths; ///< Set of currently imported paths, used to avoid recursion.
        static std::vector<std::filesystem::path> sImportDirectories; ///< Stack of import directories to properly handle adding/removing data search paths.

        /** This class is used to handle nested imports through RAII.
            It keeps a set of import paths in sImportPaths to detect recursive imports.
            It keeps a stack of import directories in sImportdirectories and updates the global data search directories.
        */
        class ScopedImport
        {
        public:
            ScopedImport(const std::filesystem::path& path)
                : mPath(path)
                , mDirectory(path.parent_path())
            {
                sImportPaths.emplace(mPath);
                sImportDirectories.push_back(mDirectory);

                // Add directory to search directories (add it to the front to make it highest priority).
                addDataDirectory(mDirectory, true);
            }
            ~ScopedImport()
            {
                auto erased = sImportPaths.erase(mPath);
                FALCOR_ASSERT(erased == 1);

                FALCOR_ASSERT(sImportDirectories.size() > 0);
                sImportDirectories.pop_back();

                // Remove script directory from search path (only if not needed by the outer importer).
                if (std::find(sImportDirectories.begin(), sImportDirectories.end(), mDirectory) == sImportDirectories.end())
                {
                    removeDataDirectory(mDirectory);
                }
            }

        private:
            std::filesystem::path mPath;
            std::filesystem::path mDirectory;
        };

        static bool isRecursiveImport(const std::filesystem::path& path)
        {
            return sImportPaths.find(path) != sImportPaths.end();
        }
    }

    void PythonImporter::import(const std::filesystem::path& path, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict)
    {
        if (!instances.empty())
        {
            throw ImporterError(path, "Python importer does not support instancing.");
        }

        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            throw ImporterError(path, "File not found.");
        }

        if (isRecursiveImport(path))
        {
            throw ImporterError(path, "Scene is imported recursively.");
        }

        // Load the script file
        const std::string script = readFile(fullPath);

        // Check for legacy .pyscene file format.
        if (auto sceneFile = parseLegacyHeader(script))
        {
            throw ImporterError(path, "Python scene file is using old header comment syntax. Use the new 'sceneBuilder' object instead.");
        }

        // Keep track of this import and add script directory to data search directories.
        // We use RAII here to make sure the scope is properly removed when throwing an exception.
        ScopedImport scopedImport(fullPath);

        // Execute script.
        try
        {
            Scripting::Context context;
            context.setObject("sceneBuilder", &builder);
            Scripting::runScript("from falcor import *", context);
            Scripting::runScriptFromFile(fullPath, context);
        }
        catch (const std::exception& e)
        {
            throw ImporterError(path, fmt::format("Failed to run python scene script: {}", e.what()));
        }
    }

    FALCOR_REGISTER_IMPORTER(
        PythonImporter,
        Importer::ExtensionList({
            "pyscene"
        })
    )
}
