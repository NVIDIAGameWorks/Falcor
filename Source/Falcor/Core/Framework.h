/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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

#ifdef _MSC_VER
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#endif

#ifdef _WIN32
#define NOMINMAX
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#define _USE_MATH_DEFINES
#include <math.h>
#include <stdint.h>
#include <memory>
#include <iostream>
#include <locale>
#include <codecvt>
#include <string>
#include <string_view>
#include <vector>
#include <algorithm>
#include <fmt/format.h> // TODO: Replace with <format> when switching to C++20

// Define DLL export/import
#ifdef _MSC_VER
#define FALCOR_API_EXPORT __declspec(dllexport)
#define FALCOR_API_IMPORT __declspec(dllimport)
#elif defined(__GNUC__) // _MSC_VER
#define FALCOR_API_EXPORT __attribute__ ((visibility ("default")))
#define FALCOR_API_IMPORT extern
#endif // _MSC_VER

#ifdef FALCOR_DLL
#define FALCOR_API FALCOR_API_EXPORT
#else   // FALCOR_DLL
#define FALCOR_API FALCOR_API_IMPORT
#endif // FALCOR_DLL

#include "Core/FalcorConfig.h"
#include "Core/ErrorHandling.h"
#include "Core/Errors.h"

// Define offsetof macro compatible with <cstddef> if not already defined.
#ifndef offsetof
#define offsetof(s, m) (size_t)( (ptrdiff_t)&reinterpret_cast<const volatile char&>((((s *)0)->m)) )
#endif

#ifdef _DEBUG

#define FALCOR_ASSERT(a)\
    if (!(a)) {\
        std::string str = "assertion failed(" + std::string(#a) + ")\nFile " + __FILE__ + ", line " + std::to_string(__LINE__);\
        Falcor::reportFatalError(str);\
    }

#else // _DEBUG

#define FALCOR_ASSERT(a) {}

#endif // _DEBUG

#define FALCOR_UNREACHABLE() FALCOR_ASSERT(false)

#define FALCOR_STRINGIZE(a) #a
#define FALCOR_CONCAT_STRINGS_(a, b) a##b
#define FALCOR_CONCAT_STRINGS(a, b) FALCOR_CONCAT_STRINGS_(a, b)


#if defined(_MSC_VER)
// Enable Windows visual styles
#pragma comment(linker,"/manifestdependency:\"type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")
#define FALCOR_FORCEINLINE __forceinline
using DllHandle = HMODULE;
#elif defined(__GNUC__)
#define FALCOR_FORCEINLINE __attribute__((always_inline))
using DllHandle = void*;
#endif

namespace Falcor
{
#define FALCOR_ENUM_CLASS_OPERATORS(e_) \
    inline e_ operator& (e_ a, e_ b) { return static_cast<e_>(static_cast<int>(a)& static_cast<int>(b)); } \
    inline e_ operator| (e_ a, e_ b) { return static_cast<e_>(static_cast<int>(a)| static_cast<int>(b)); } \
    inline e_& operator|= (e_& a, e_ b) { a = a | b; return a; }; \
    inline e_& operator&= (e_& a, e_ b) { a = a & b; return a; }; \
    inline e_  operator~ (e_ a) { return static_cast<e_>(~static_cast<int>(a)); } \
    inline bool is_set(e_ val, e_ flag) { return (val & flag) != static_cast<e_>(0); } \
    inline void flip_bit(e_& val, e_ flag) { val = is_set(val, flag) ? (val & (~flag)) : (val | flag); }

    /*!
    *  \addtogroup Falcor
    *  @{
    */

    /** Falcor shader types
    */
    enum class ShaderType
    {
        Vertex,         ///< Vertex shader
        Pixel,          ///< Pixel shader
        Geometry,       ///< Geometry shader
        Hull,           ///< Hull shader (AKA Tessellation control shader)
        Domain,         ///< Domain shader (AKA Tessellation evaluation shader)
        Compute,        ///< Compute shader

#if defined(FALCOR_D3D12) || defined(FALCOR_GFX)
        RayGeneration,  ///< Ray generation shader
        Intersection,   ///< Intersection shader
        AnyHit,         ///< Any hit shader
        ClosestHit,     ///< Closest hit shader
        Miss,           ///< Miss shader
        Callable,       ///< Callable shader
#endif
        Count           ///< Shader Type count
    };

    /** Converts ShaderType enum elements to a string.
        \param[in] type Type to convert to string
        \return Shader type as a string
    */
    inline const std::string to_string(ShaderType Type)
    {
        switch(Type)
        {
        case ShaderType::Vertex:
            return "vertex";
        case ShaderType::Pixel:
            return "pixel";
        case ShaderType::Hull:
            return "hull";
        case ShaderType::Domain:
            return "domain";
        case ShaderType::Geometry:
            return "geometry";
        case ShaderType::Compute:
            return "compute";
#ifdef FALCOR_D3D12
        case ShaderType::RayGeneration:
            return "raygeneration";
        case ShaderType::Intersection:
            return "intersection";
        case ShaderType::AnyHit:
            return "anyhit";
        case ShaderType::ClosestHit:
            return "closesthit";
        case ShaderType::Miss:
            return "miss";
        case ShaderType::Callable:
            return "callable";
#endif
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
    }

    /** Framebuffer target flags. Used for clears and copy operations
    */
    enum class FboAttachmentType
    {
        None    = 0,    ///< Nothing. Here just for completeness
        Color   = 1,    ///< Operate on the color buffer.
        Depth   = 2,    ///< Operate on the the depth buffer.
        Stencil = 4,    ///< Operate on the the stencil buffer.

        All = Color | Depth | Stencil ///< Operate on all targets
    };

    FALCOR_ENUM_CLASS_OPERATORS(FboAttachmentType);


    enum class ComparisonFunc
    {
        Disabled,       ///< Comparison is disabled
        Never,          ///< Comparison always fails
        Always,         ///< Comparison always succeeds
        Less,           ///< Passes if source is less than the destination
        Equal,          ///< Passes if source is equal to the destination
        NotEqual,       ///< Passes if source is not equal to the destination
        LessEqual,      ///< Passes if source is less than or equal to the destination
        Greater,        ///< Passes if source is greater than to the destination
        GreaterEqual,   ///< Passes if source is greater than or equal to the destination
    };

    /** Flags indicating what hot-reloadable resources have changed
    */
    enum class HotReloadFlags
    {
        None    = 0,    ///< Nothing. Here just for completeness
        Program = 1,    ///< Programs (shaders)
    };

    FALCOR_ENUM_CLASS_OPERATORS(HotReloadFlags);

    /** Clamps a value within a range.
        \param[in] val Value to clamp
        \param[in] minVal Low end to clamp to
        \param[in] maxVal High end to clamp to
        \return Result
    */
    template<typename T>
    inline T clamp(const T& val, const T& minVal, const T& maxVal)
    {
        return std::min(std::max(val, minVal), maxVal);
    }

    /** Returns whether an integer number is a power of two.
    */
    template<typename T>
    inline typename std::enable_if<std::is_integral<T>::value, bool>::type isPowerOf2(T a)
    {
        return (a & (a - (T)1)) == 0;
    }

    template <typename T>
    inline T div_round_up(T a, T b) { return (a + b - (T)1) / b; }

    /** Helper class to check if a class has a vtable.
        Usage: has_vtable<MyClass>::value is true if vtable exists, false otherwise.
    */
    template<class T>
    struct has_vtable
    {
        class derived : public T
        {
            virtual void force_the_vtable() {}
        };
        enum { value = (sizeof(T) == sizeof(derived)) };
    };

    /** Helper to get the size of a C array variable.
    */
    template<typename T, size_t N>
    constexpr size_t arraysize(T(&)[N]) { return N; }

    /** Helper to align a value to a given alignment.
    */
    template<typename T>
    constexpr T align_to(T alignment, T value)
    {
        static_assert(std::is_integral<T>::value, "Integral type required.");
        return ((value + alignment - T(1)) / alignment) * alignment;
    }

    /** Helper to delete and assign nullptr.
    */
    template<typename T>
    void safe_delete(T*& ptr)
    {
        delete ptr;
        ptr = nullptr;
    }

    /*! @} */
}

#include "Utils/Math/Vector.h"
#include "Utils/Math/Float16.h"

#if defined(FALCOR_D3D12)
#include "Core/API/D3D12/FalcorD3D12.h"
#elif defined(FALCOR_GFX)
#include "Core/API/GFX/FalcorGFX.h"
#else
#error Undefined falcor backend. Make sure that a backend is selected in "FalcorConfig.h"
#endif

#include "Core/Platform/OS.h"
#include "Utils/Logger.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Scripting/Scripting.h"

#if FALCOR_ENABLE_NVAPI
#include "nvapi.h"
#pragma comment(lib, "nvapi64.lib")
#endif
