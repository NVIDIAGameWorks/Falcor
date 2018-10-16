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
#include <iostream>
#include "Utils/Logger.h"
#include "Utils/Scripting/Scripting.h"

#define GLM_FORCE_CTOR_INIT
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
using namespace glm;

// Define DLL export/import
#ifdef _MSC_VER
#define falcorexport __declspec(dllexport)
#define falcorimport __declspec(dllimport)
#else  // _MSC_VER
#define falcorexport __attribute__ ((visibility ("default")))
#define falcorimport extern
#endif // _MSC_VER

#ifdef BUILDING_SHARED_DLL
#define dlldecl falcorexport
#else   // BUILDING_SHARED_DLL
#define dlldecl falcorimport
#endif // BUILDING_SHARED_DLL

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
#define concat_strings_(a, b) a##b
#define concat_strings(a, b) concat_strings_(a, b)

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

#ifdef FALCOR_D3D12
        RayGeneration,  ///< Ray generation shader
        Intersection,   ///< Intersection shader
        AnyHit,         ///< Any hit shader
        ClosestHit,     ///< Closest hit shader
        Miss,           ///< Miss shader
        Callable,       ///< Callable shader
#endif
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
            should_not_get_here();
            return "";
        }
    }


#define compare_str(a) case ComparisonFunc::a: return #a
    inline std::string to_string(ComparisonFunc f)
    {
        switch (f)
        {
            compare_str(Disabled);
            compare_str(LessEqual);
            compare_str(GreaterEqual);
            compare_str(Less);
            compare_str(Greater);
            compare_str(Equal);
            compare_str(NotEqual);
            compare_str(Always);
            compare_str(Never);
        default: should_not_get_here(); return "";
        }
    }
#undef compare_str
}

#if defined(_MSC_VER)
#define deprecate(_ver_, _msg_) __declspec(deprecated("This function is deprecated and will be removed in Falcor " ##  _ver_ ## ". " ## _msg_))
#define forceinline __forceinline
using DllHandle = HMODULE;
#else
#define deprecate(_ver_, _msg_) 
#define forceinline __attribute__((always_inline))
using DllHandle = void*;
#endif

#include "Utils/Platform/OS.h"
#include "Utils/Profiler.h"

#if (_ENABLE_NVAPI == true)
#include "nvapi.h"
#pragma comment(lib, "nvapi64.lib")
#endif