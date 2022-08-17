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
#include "Core/Macros.h"
#include "Utils/Math/Vector.h"
#include <cstdint>

namespace Falcor
{
    /** Raytracing pipeline flags.
    */
    enum class RtPipelineFlags : uint32_t
    {
        None = 0,
        SkipTriangles = 0x1,
        SkipProceduralPrimitives = 0x2,
    };
    FALCOR_ENUM_CLASS_OPERATORS(RtPipelineFlags);

    /** Raytracing axis-aligned bounding box.
    */
    struct RtAABB
    {
        float3 min;
        float3 max;
    };

    /** Flags passed to TraceRay(). These must match the device side.
    */
    enum class RayFlags : uint32_t
    {
        None,
        ForceOpaque = 0x1,
        ForceNonOpaque = 0x2,
        AcceptFirstHitAndEndSearch = 0x4,
        SkipClosestHitShader = 0x8,
        CullBackFacingTriangles = 0x10,
        CullFrontFacingTriangles = 0x20,
        CullOpaque = 0x40,
        CullNonOpaque = 0x80,
        SkipTriangles = 0x100,
        SkipProceduralPrimitives = 0x200,
    };
    FALCOR_ENUM_CLASS_OPERATORS(RayFlags);

    // Maximum raytracing attribute size.
    inline constexpr uint32_t getRaytracingMaxAttributeSize() { return 32; }
}
