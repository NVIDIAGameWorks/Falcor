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
#include "Core/Macros.h"
#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include <filesystem>
#include <string>

/// ImporterError split from the Importer.h file to allow using it without bringing in pybind11 types (via SceneBuilder)

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

        ImporterError(const std::filesystem::path& path, std::string_view what)
            : Exception(what)
            , mpPath(std::make_shared<std::filesystem::path>(path))
        {}

        template<typename... Args>
        explicit ImporterError(const std::filesystem::path& path, fmt::format_string<Args...> format, Args&&... args)
            : ImporterError(path, fmt::format(format, std::forward<Args>(args)...))
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
}
