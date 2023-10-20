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

#include "Core/Enum.h"

namespace Falcor
{

enum class ShaderModel : uint32_t
{
    Unknown = 0,
    SM6_0 = 60,
    SM6_1 = 61,
    SM6_2 = 62,
    SM6_3 = 63,
    SM6_4 = 64,
    SM6_5 = 65,
    SM6_6 = 66,
    SM6_7 = 67,
};
FALCOR_ENUM_INFO(
    ShaderModel,
    {
        {ShaderModel::Unknown, "Unknown"},
        {ShaderModel::SM6_0, "SM6_0"},
        {ShaderModel::SM6_1, "SM6_1"},
        {ShaderModel::SM6_2, "SM6_2"},
        {ShaderModel::SM6_3, "SM6_3"},
        {ShaderModel::SM6_4, "SM6_4"},
        {ShaderModel::SM6_5, "SM6_5"},
        {ShaderModel::SM6_6, "SM6_6"},
        {ShaderModel::SM6_7, "SM6_7"},
    }
);
FALCOR_ENUM_REGISTER(ShaderModel);

inline uint32_t getShaderModelMajorVersion(ShaderModel sm)
{
    return uint32_t(sm) / 10;
}
inline uint32_t getShaderModelMinorVersion(ShaderModel sm)
{
    return uint32_t(sm) % 10;
}

/**
 * Falcor shader types
 */
enum class ShaderType
{
    Vertex,        ///< Vertex shader
    Pixel,         ///< Pixel shader
    Geometry,      ///< Geometry shader
    Hull,          ///< Hull shader (AKA Tessellation control shader)
    Domain,        ///< Domain shader (AKA Tessellation evaluation shader)
    Compute,       ///< Compute shader
    RayGeneration, ///< Ray generation shader
    Intersection,  ///< Intersection shader
    AnyHit,        ///< Any hit shader
    ClosestHit,    ///< Closest hit shader
    Miss,          ///< Miss shader
    Callable,      ///< Callable shader
    Count          ///< Shader Type count
};
FALCOR_ENUM_INFO(
    ShaderType,
    {
        {ShaderType::Vertex, "Vertex"},
        {ShaderType::Pixel, "Pixel"},
        {ShaderType::Geometry, "Geometry"},
        {ShaderType::Hull, "Hull"},
        {ShaderType::Domain, "Domain"},
        {ShaderType::Compute, "Compute"},
        {ShaderType::RayGeneration, "RayGeneration"},
        {ShaderType::Intersection, "Intersection"},
        {ShaderType::AnyHit, "AnyHit"},
        {ShaderType::ClosestHit, "ClosestHit"},
        {ShaderType::Miss, "Miss"},
        {ShaderType::Callable, "Callable"},
    }
);
FALCOR_ENUM_REGISTER(ShaderType);

enum class DataType
{
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float16,
    float32,
    float64,
};
FALCOR_ENUM_INFO(
    DataType,
    {
        {DataType::int8, "int8"},
        {DataType::int16, "int16"},
        {DataType::int32, "int32"},
        {DataType::int64, "int64"},
        {DataType::uint8, "uint8"},
        {DataType::uint16, "uint16"},
        {DataType::uint32, "uint32"},
        {DataType::uint64, "uint64"},
        {DataType::float16, "float16"},
        {DataType::float32, "float32"},
        {DataType::float64, "float64"},
    }
);
FALCOR_ENUM_REGISTER(DataType);

enum class ComparisonFunc
{
    Disabled,     ///< Comparison is disabled
    Never,        ///< Comparison always fails
    Always,       ///< Comparison always succeeds
    Less,         ///< Passes if source is less than the destination
    Equal,        ///< Passes if source is equal to the destination
    NotEqual,     ///< Passes if source is not equal to the destination
    LessEqual,    ///< Passes if source is less than or equal to the destination
    Greater,      ///< Passes if source is greater than to the destination
    GreaterEqual, ///< Passes if source is greater than or equal to the destination
};

FALCOR_ENUM_INFO(
    ComparisonFunc,
    {
        {ComparisonFunc::Disabled, "Disabled"},
        {ComparisonFunc::Never, "Never"},
        {ComparisonFunc::Always, "Always"},
        {ComparisonFunc::Less, "Less"},
        {ComparisonFunc::Equal, "Equal"},
        {ComparisonFunc::NotEqual, "NotEqual"},
        {ComparisonFunc::LessEqual, "LessEqual"},
        {ComparisonFunc::Greater, "Greater"},
        {ComparisonFunc::GreaterEqual, "GreaterEqual"},
    }
);
FALCOR_ENUM_REGISTER(ComparisonFunc);

} // namespace Falcor
