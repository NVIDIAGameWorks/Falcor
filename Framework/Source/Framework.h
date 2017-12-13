/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/
#pragma once
#include "FalcorConfig.h"
#include <stdint.h>
#include <memory>
#include "glm/glm.hpp"
#include <iostream>
#include "Utils/Logger.h"

using namespace glm;

#ifndef arraysize
#define arraysize(a) (sizeof(a)/sizeof(a[0]))
#endif
#ifndef offsetof
#define offsetof(s, m) (size_t)( (ptrdiff_t)&reinterpret_cast<const volatile char&>((((s *)0)->m)) )
#endif

#ifdef assert
#undef assert
#endif

#ifdef _DEBUG
#define assert(a)\
    if (!(a)) {\
        std::string str = "assertion failed(" + std::string(#a) + ")\nFile " + __FILE__ + ", line " + std::to_string(__LINE__);\
        Falcor::logError(str);\
    }

#define should_not_get_here() assert(false);

#else // _DEBUG

#ifdef _AUTOTESTING
#define assert(a) if (!(a)) throw std::exception("Assertion Failure");
#else // _AUTOTESTING
#define assert(a) ((void)(a))
#endif // _AUTOTESTING

#ifdef _MSC_VER
#define should_not_get_here() __assume(0)
#else // _MSC_VER
#define should_not_get_here() __builtin_unreachable()
#endif // _MSC_VER

#endif // _DEBUG

#define safe_delete(_a) {delete _a; _a = nullptr;}
#define safe_delete_array(_a) {delete[] _a; _a = nullptr;}
#define align_to(_alignment, _val) (((_val + _alignment - 1) / _alignment) * _alignment)

#if defined(_MSC_VER)
#define FALCOR_DEPRECATED(MESSAGE) __declspec(deprecated(MESSAGE))
#define forceinline __forceinline
#else
// TODO: add cases for clang/gcc when/if needed
#define FALCOR_DEPRECATED(MESSAGE) /* emtpy */
#define forceinline __attribute__((always_inline))
#endif

namespace Falcor
{
#define enum_class_operators(e_) inline e_ operator& (e_ a, e_ b){return static_cast<e_>(static_cast<int>(a)& static_cast<int>(b));}  \
    inline e_ operator| (e_ a, e_ b){return static_cast<e_>(static_cast<int>(a)| static_cast<int>(b));} \
    inline e_& operator|= (e_& a, e_ b){a = a | b; return a;};  \
    inline e_& operator&= (e_& a, e_ b) { a = a & b; return a; };   \
    inline e_  operator~ (e_ a) { return static_cast<e_>(~static_cast<int>(a));}   \
    inline bool is_set(e_ val, e_ flag) { return (val & flag) != (e_)0;}

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

        Extended,       ///< An extended, non-standard shader type

        Count           ///< Shader Type count
    };


    /** Shading languages. Used for shader cross-compilation.
    */
    enum class ShadingLanguage
    {
        Unknown,        ///< Unknown language (e.g., for a plain .h file)
        GLSL,           ///< OpenGL Shading Language (GLSL)
        VulkanGLSL,     ///< GLSL for Vulkan
        HLSL,           ///< High-Level Shading Language
        Slang,          ///< Slang shading language
    };

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

    enum_class_operators(FboAttachmentType);

    /** Clamps a value within a range.
        \param[in] val Value to clamp
        \param[in] minVal Low end to clamp to
        \param[in] maxVal High end to clamp to
        \return Result
    */
    template<typename T>
    inline T clamp(const T& val, const T& minVal, const T& maxVal)
    {
        return min(max(val, minVal), maxVal);
    }

    /** Returns whether a number is a power of two
    */
    template<typename T>
    inline bool isPowerOf2(T a)
    {
        uint64_t t = (uint64_t)a;
        return (t & (t - 1)) == 0;
    }

    /*! @} */


    // This is a helper class which should be used in case a class derives from a base class which derives from enable_shared_from_this
    // If Derived will also inherit enable_shared_from_this, it will cause multiple inheritance from enable_shared_from_this, which results in a runtime errors because we have 2 copies of the WeakPtr inside shared_ptr
    template<typename Base, typename Derived>
    class inherit_shared_from_this
    {
    public:
        typename std::shared_ptr<Derived> shared_from_this()
        {
            Base* pBase = static_cast<Derived*>(this);
            std::shared_ptr<Base> pShared = pBase->shared_from_this();
            return std::static_pointer_cast<Derived>(pShared);
        }

        typename std::shared_ptr<const Derived> shared_from_this() const
        {
            const Base* pBase = static_cast<const Derived*>(this);
            std::shared_ptr<const Base> pShared = pBase->shared_from_this();
            return std::static_pointer_cast<const Derived>(pShared);
        }
    };
}

#if defined(FALCOR_D3D12)
#include "API/D3D12/FalcorD3D12.h"
#elif defined(FALCOR_VK)
#include "API/Vulkan/FalcorVK.h"
#else
#error Undefined falcor backend. Make sure that a backend is selected in "FalcorConfig.h"
#endif

#if defined(FALCOR_D3D12) || defined(FALCOR_VK)
#define FALCOR_LOW_LEVEL_API
#endif

namespace Falcor
{
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
        default:
            should_not_get_here();
            return "";
        }
    }
}

#include "Utils/Platform/OS.h"
#include "Utils/Profiler.h"

#if (_ENABLE_NVAPI == true)
#include "nvapi.h"
#pragma comment(lib, "nvapi64.lib")
#endif
