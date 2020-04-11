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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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

    private:
        bool error(const std::string& msg);

        SceneBuilder& mBuilder;
        std::string mFilename;
        std::string mDirectory;
    };

    bool PythonImporterImpl::error(const std::string& msg)
    {
        logError("Error when parsing scene file \"" + mFilename + "\".\n" + msg);
        return false;
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

            // Get filename of referenced scene from first line "# [scene filename]"
            size_t endOfFirstLine = script.find_first_of("\r\n");
            if (script.length() < 2 || script[0] != '#' || script[1] != ' ' || endOfFirstLine == std::string::npos)
            {
                return error("Script file is missing header with reference to scene file.");
            }

            addDataDirectory(directory);

            // Load referenced scene
            const std::string sceneFile = script.substr(2, endOfFirstLine - 2);
            mBuilder.import(sceneFile.c_str());

            // Execute scene script
            Scripting::Context context;
            context.setObject("scene", mBuilder.getScene());
            Scripting::runScriptFromFile(fullpath, context);

            removeDataDirectory(directory);
            return true;
        }
        else
        {
            return false;
        }
    }

    bool PythonImporter::import(const std::string& filename, SceneBuilder& builder)
    {
        PythonImporterImpl importer(builder);
        return importer.load(filename);
    }
}
