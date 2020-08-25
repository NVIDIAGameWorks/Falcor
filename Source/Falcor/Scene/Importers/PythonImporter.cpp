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

namespace Falcor
{
    class PythonImporterImpl
    {
    public:
        PythonImporterImpl(SceneBuilder& builder) : mBuilder(builder) {}
        bool load(const std::string& filename);
        bool importScene(std::string& filename, const pybind11::dict& dict);
    private:
        Scripting::Context mScriptingContext;
        SceneBuilder& mBuilder;
    };

    bool PythonImporterImpl::importScene(std::string& filename, const pybind11::dict& dict)
    {
        bool success = true;

        std::string extension = std::filesystem::path(filename).extension().string();
        if (extension == ".pyscene")
        {
            logError("Python scene files cannot be imported from python scene files.");
            success = false;
        }
        else
        {
            SceneBuilder::InstanceMatrices mats;
            success = mBuilder.import(filename, mats, Dictionary(dict));

            if (success)
            {
                if (mScriptingContext.containsObject("scene"))
                {
                    // Warn if a scene had previously been imported by this script
                    logWarning("More than one scene loaded from python script. Discarding previously loaded scene.");
                }
                mScriptingContext.setObject("scene", mBuilder.getScene());
            }
        }
        return success;
    }

    bool PythonImporterImpl::load(const std::string& filename)
    {
        std::string fullpath;

        if (findFileInDataDirectories(filename, fullpath))
        {
            // Get the directory of the script file
            const std::string directory = fullpath.substr(0, fullpath.find_last_of("/\\"));

            // Load the script file
            const std::string script = removeLeadingWhitespaces(readFile(fullpath));

            addDataDirectory(directory);

            bool success = true;

            // Get filename of referenced scene from first line "# filename.{fbx,fscene}", if any
            size_t endOfFirstLine = script.find_first_of("\r\n");
            if (script.length() >= 2 && script[0] == '#' && script[1] == ' ' && endOfFirstLine != std::string::npos)
            {
                const std::string sceneFile = script.substr(2, endOfFirstLine - 2);
                std::string extension = std::filesystem::path(sceneFile).extension().string();
                if (extension != ".pyscene")
                {
                    // Load referenced scene
                    success = mBuilder.import(sceneFile.c_str());
                    if (success)
                    {
                        mScriptingContext.setObject("scene", mBuilder.getScene());
                    }
                }
            }

            if (success)
            {
                // Execute scene script.
                mScriptingContext.setObject("importer", this);
                Scripting::runScriptFromFile(fullpath, mScriptingContext);
            }

            removeDataDirectory(directory);
            return success;
        }
        else
        {
            logError("Error when loading scene file '" + filename + "'. File not found.");
            return false;
        }
    }

    bool PythonImporter::import(const std::string& filename, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict)
    {
        if (!instances.empty()) logWarning("Python importer does not support instancing.");

        PythonImporterImpl importer(builder);
        return importer.load(filename);
    }

    SCRIPT_BINDING(PythonImporterImpl)
    {
        pybind11::class_<PythonImporterImpl> importer(m, "PythonImporterImpl");
        importer.def("importScene", &PythonImporterImpl::importScene, "filename"_a, "dictionary"_a = pybind11::dict());
    }

    REGISTER_IMPORTER(
        PythonImporter,
        Importer::ExtensionList({
            "pyscene"
        })
    )
}
