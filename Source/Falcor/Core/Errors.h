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
#include "Macros.h"
#include <fmt/format.h> // TODO C++20: Replace with <format>
#include <exception>
#include <memory>
#include <string>
#include <string_view>

namespace Falcor
{

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4275) // allow dllexport on classes dervied from STL
#endif

    /** Base class for all Falcor exceptions.
    */
    class FALCOR_API Exception : public std::exception
    {
    public:
        Exception() noexcept
        {}

        Exception(const char* what)
            : mpWhat(std::make_shared<std::string>(what))
        {}

        Exception(const std::string& what)
            : Exception(what.c_str())
        {}

        template<typename... Args>
        explicit Exception(const std::string_view format, Args&&... args)
            : Exception(fmt::vformat(format, fmt::make_format_args(std::forward<Args>(args)...)).c_str())
        {}

        Exception(const Exception& other) noexcept
        {
            mpWhat = other.mpWhat;
        }

        virtual ~Exception() override
        {}

        virtual const char* what() const noexcept override
        {
            return mpWhat ? mpWhat->c_str() : "";
        }

    protected:
        // Message is stored as a reference counted string in order to allow copy constructor to be noexcept.
        std::shared_ptr<std::string> mpWhat;
    };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

    /** Exception to be thrown when an error happens at runtime.
    */
    class FALCOR_API RuntimeError : public Exception
    {
    public:
        RuntimeError() noexcept
        {}

        RuntimeError(const char* what)
            : Exception(what)
        {}

        RuntimeError(const std::string& what)
            : Exception(what)
        {}

        template<typename... Args>
        explicit RuntimeError(const std::string_view format, Args&&... args)
            : Exception(format, std::forward<Args>(args)...)
        {}

        RuntimeError(const RuntimeError& other) noexcept
        {
            mpWhat = other.mpWhat;
        }

        virtual ~RuntimeError() override
        {}
    };

    /** Exception to be thrown when a function argument has an invalid value.
    */
    class FALCOR_API ArgumentError : public Exception
    {
    public:
        ArgumentError() noexcept
        {}

        ArgumentError(const char* what)
            : Exception(what)
        {}

        ArgumentError(const std::string& what)
            : Exception(what)
        {}

        template<typename... Args>
        explicit ArgumentError(const std::string_view format, Args&&... args)
            : Exception(format, std::forward<Args>(args)...)
        {}

        virtual ~ArgumentError() override
        {}

        ArgumentError(const ArgumentError& other) noexcept
        {
            mpWhat = other.mpWhat;
        }
    };

    /** Check that an invariant holds and throw a RuntimeError if it doesn't.
        \param[in] condition Invariant condition that must hold.
        \param[in] format Format string.
        \param[in] ... Arguments.
    */
    template<typename... Args>
    void checkInvariant(bool condition, const std::string_view format, Args&&... args)
    {
        if (!condition) throw RuntimeError(format, std::forward<Args>(args)...);
    }

    /** Check if a function argument meets some condition and throws an ArgumentError if it doesn't.
        \param[in] condition Argument condition that must hold.
        \param[in] format Format string.
        \param[in] ... Arguments.
    */
    template<typename... Args>
    void checkArgument(bool condition, const std::string_view format, Args&&... args)
    {
        if (!condition) throw ArgumentError(format, std::forward<Args>(args)...);
    }
}
