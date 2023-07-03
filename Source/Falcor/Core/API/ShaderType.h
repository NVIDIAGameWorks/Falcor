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
#include "Core/Assert.h"

#include <string>

namespace Falcor
{
/**
 * Falcor shader types
 */
enum class ShaderType
{
    Vertex,   ///< Vertex shader
    Pixel,    ///< Pixel shader
    Geometry, ///< Geometry shader
    Hull,     ///< Hull shader (AKA Tessellation control shader)
    Domain,   ///< Domain shader (AKA Tessellation evaluation shader)
    Compute,  ///< Compute shader

    RayGeneration, ///< Ray generation shader
    Intersection,  ///< Intersection shader
    AnyHit,        ///< Any hit shader
    ClosestHit,    ///< Closest hit shader
    Miss,          ///< Miss shader
    Callable,      ///< Callable shader
    Count          ///< Shader Type count
};

/**
 * Converts ShaderType enum elements to a string.
 * @param[in] type Type to convert to string
 * @return Shader type as a string
 */
inline const std::string to_string(ShaderType type)
{
    switch (type)
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
    default:
        FALCOR_UNREACHABLE();
        return "";
    }
}

} // namespace Falcor
