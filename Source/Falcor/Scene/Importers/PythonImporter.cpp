/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "PythonImporter.h"
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
    }

    bool PythonImporter::import(const std::string& filename, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict)
    {
        bool success = false;

        if (!instances.empty()) logWarning("Python importer does not support instancing.");

        std::string fullpath;

        if (findFileInDataDirectories(filename, fullpath))
        {
            // Add script directory to search paths (add it to the front to make it highest priority).
            const std::string directory = getDirectoryFromFile(fullpath);
            addDataDirectory(directory, true);

            // Load the script file
            const std::string script = readFile(fullpath);

            // Check for legacy .pyscene file format.
            if (auto sceneFile = parseLegacyHeader(script))
            {
                logError("Python scene file '" + fullpath + "' is using old header comment syntax. Use the new 'sceneBuilder' object instead.");
            }
            else
            {
                Scripting::Context context;
                context.setObject("sceneBuilder", &builder);
                Scripting::runScript("from falcor import *", context);
                Scripting::runScriptFromFile(fullpath, context);
                success = true;
            }

            // Remove script directory from search path.
            removeDataDirectory(directory);
        }
        else
        {
            logError("Error when loading scene file '" + filename + "'. File not found.");
        }

        return success;
    }

    REGISTER_IMPORTER(
        PythonImporter,
        Importer::ExtensionList({
            "pyscene"
        })
    )
}
