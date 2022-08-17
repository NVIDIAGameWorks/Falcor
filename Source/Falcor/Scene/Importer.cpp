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
#include "Importer.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    namespace
    {
        static std::vector<Importer::Desc> sImporters;
        static std::unordered_map<std::string, Importer::ImportFunction> sImportFunctions;
        static FileDialogFilterVec sFileExtensionsFilters;
    }

    const FileDialogFilterVec& Importer::getFileExtensionFilters()
    {
        return sFileExtensionsFilters;
    }

    void Importer::import(const std::filesystem::path& path, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict)
    {
        auto ext = getExtensionFromPath(path);
        auto it = sImportFunctions.find(ext);
        if (it == sImportFunctions.end())
        {
            throw ImporterError(path, "Unknown file extension.");
        }
        it->second(path, builder, instances, dict);
    }

    void Importer::registerImporter(const Desc& desc)
    {
        sImporters.push_back(desc);

        for (const auto& ext : desc.extensions)
        {
            FALCOR_ASSERT(sImportFunctions.find(ext) == sImportFunctions.end());
            sImportFunctions[ext] = desc.import;
            sFileExtensionsFilters.push_back(ext);
        }
    }

    FALCOR_SCRIPT_BINDING(Importer)
    {
        pybind11::register_exception<ImporterError>(m, "ImporterError");
    }
}
