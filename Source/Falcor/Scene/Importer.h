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
#pragma once
#include "SceneBuilder.h"
#include "Core/Macros.h"
#include "Core/Errors.h"
#include "Core/Platform/OS.h"
#include "Utils/Scripting/Dictionary.h"
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Falcor
{
    /** Exception thrown during scene import.
        Holds the path of the imported asset and a description of the exception.
    */
    class FALCOR_API ImporterError : public Exception
    {
    public:
        ImporterError() noexcept
        {}

        ImporterError(const std::filesystem::path& path, const char* what)
            : Exception(what)
            , mpPath(std::make_shared<std::filesystem::path>(path))
        {}

        ImporterError(const std::filesystem::path& path, const std::string& what)
            : ImporterError(path, what.c_str())
        {}

        template<typename... Args>
        explicit ImporterError(const std::filesystem::path& path, const std::string_view format, Args&&... args)
            : ImporterError(path, fmt::vformat(format, fmt::make_format_args(std::forward<Args>(args)...)).c_str())
        {}

        virtual ~ImporterError() override
        {}

        ImporterError(const ImporterError& other) noexcept
        {
            mpWhat = other.mpWhat;
            mpPath = other.mpPath;
        }

        const std::filesystem::path& path() const noexcept { return *mpPath; }

    private:
        std::shared_ptr<std::filesystem::path> mpPath;
    };

    /** This class is a global registry for asset importers.
        Importers are bound to a set of file extensions. This allows the right importer to
        be called when importing an asset file.
    */
    class FALCOR_API Importer
    {
    public:
        using ExtensionList = std::vector<std::string>;
        using ImportFunction = std::function<void(const std::filesystem::path& path, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict)>;

        /** Description of an importer.
        */
        struct Desc
        {
            std::string name;
            ExtensionList extensions;
            ImportFunction import;
        };

        /** Returns a list of file extensions filters for all supported file formats.
        */
        static const FileDialogFilterVec& getFileExtensionFilters();

        /** Import an asset.
            \param[in] path File path.
            \param[in] builder Scene builder.
            \param[in] instances Optional list of instance transforms.
            \param[in] dict Optional dictionary.
            Throws an ImporterError if something went wrong.
        */
        static void import(const std::filesystem::path& path, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict);

        /** Registers an importer.
            \param[in] desc Importer description.
        */
        static void registerImporter(const Desc& desc);
    };
}

#ifndef _staticlibrary
#define FALCOR_REGISTER_IMPORTER(_class, _extensions)                               \
    static struct RegisterImporter##_class {                                        \
        RegisterImporter##_class()                                                  \
        {                                                                           \
            Importer::registerImporter({#_class, _extensions, _class::import});     \
        }                                                                           \
    } gRegisterImporter##_class;
#else
#define FALCOR_REGISTER_IMPORTER(_class, _extensions) \
    static_assert(false, "Using FALCOR_REGISTER_IMPORTER() in a static-library is not supported. The C++ linker usually doesn't pull static-initializers into the EXE. " \
    "Call 'Importer::registerImporter()' yourself from code that is guarenteed to run.");
#endif
