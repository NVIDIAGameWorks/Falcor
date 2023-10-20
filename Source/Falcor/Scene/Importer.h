/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "SceneBuilder.h"
#include "ImporterError.h"
#include "Core/Macros.h"
#include "Core/Error.h"
#include "Core/Plugin.h"
#include "Core/Platform/OS.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Falcor
{
    /** Base class for importers.
        Importers are bound to a set of file extensions. This allows the right importer to
        be called when importing an asset file.
    */
    class FALCOR_API Importer
    {
    public:
        using PluginCreate = std::function<std::unique_ptr<Importer>()>;
        struct PluginInfo
        {
            std::string desc; ///< Importer description.
            std::vector<std::string> extensions; ///< List of handled file extensions.
        };

        FALCOR_PLUGIN_BASE_CLASS(Importer);

        virtual ~Importer() {}

        /** Import a scene.
            \param[in] path File path.
            \param[in] builder Scene builder.
            \param[in] dict Optional dictionary.
            Throws an ImporterError if something went wrong.
        */
        virtual void importScene(const std::filesystem::path& path, SceneBuilder& builder, const std::map<std::string, std::string>& materialToShortName) = 0;

        /** Import a scene from memory.
            \param[in] buffer Memory buffer.
            \param[in] byteSize Size in bytes of memory buffer.
            \param[in] extension File extension for the format the scene is stored in.
            \param[in] builder Scene builder.
            \param[in] dict Optional dictionary.
            Throws an ImporterError if something went wrong.
        */
        virtual void importSceneFromMemory(const void* buffer, size_t byteSize, std::string_view extension, SceneBuilder& builder, const std::map<std::string, std::string>& materialToShortName);

        // Importer factory

        /** Create an importer for a file of an asset with the given file extension.
            \param extension File extension.
            \param pm Plugin manager.
            \return Returns an instance of the importer or nullptr if no compatible importer was found.
         */
        static std::unique_ptr<Importer> create(std::string_view extension, const PluginManager& pm = PluginManager::instance());

        /** Return a list of supported file extensions by the current set of loaded importer plugins.
        */
        static std::vector<std::string> getSupportedExtensions(const PluginManager& pm = PluginManager::instance());
    };
}
