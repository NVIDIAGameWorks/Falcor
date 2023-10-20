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
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class DiffMode : uint32_t
{
    Primal,            // Outputs primal images like path tracers do.
    BackwardDiff,      // Outputs gradient vectors.
    ForwardDiffDebug,  // Outputs gradient images w.r.t. a single variable for debugging.
    BackwardDiffDebug, // Outputs gradient images w.r.t. a single variable for debugging.
};

FALCOR_ENUM_INFO(
    DiffMode,
    {
        { DiffMode::Primal, "Primal" },
        { DiffMode::BackwardDiff, "BackwardDiff" },
        { DiffMode::ForwardDiffDebug, "ForwardDiffDebug" },
        { DiffMode::BackwardDiffDebug, "BackwardDiffDebug" },
    }
);
FALCOR_ENUM_REGISTER(DiffMode);

// We will probably add more gradient types in the future.
enum class GradientType : uint32_t
{
    Material,
    Geometry,
    Count,
};

FALCOR_ENUM_INFO(
    GradientType,
    {
        { GradientType::Material, "Material" },
        { GradientType::Geometry, "Geometry" },
    }
);
FALCOR_ENUM_REGISTER(GradientType);

enum class GradientAggregateMode : uint32_t
{
    Direct,
    HashGrid,
};

// For debugging differentiable path tracers by visualizing gradient images.

enum class DiffVariableType : uint32_t
{
    None,
    Material,
    GeometryTranslation,
};

FALCOR_ENUM_INFO(
    DiffVariableType,
    {
        { DiffVariableType::None, "None" },
        { DiffVariableType::Material, "Material" },
        { DiffVariableType::GeometryTranslation, "GeometryTranslation" },
    }
);
FALCOR_ENUM_REGISTER(DiffVariableType);

struct DiffDebugParams
{
    DiffVariableType varType;
    uint2 id;
    uint32_t offset;

    float4 grad;
};

END_NAMESPACE_FALCOR
