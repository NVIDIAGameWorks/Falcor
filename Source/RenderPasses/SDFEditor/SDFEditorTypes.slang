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

/** Structs and enums for GUIPass for the SDFEditor
*/

#pragma once
#include "Utils/HostDeviceShared.slangh"

#ifdef HOST_CODE
#include "Utils/Math/AABB.h"
#include "Scene/HitInfoType.slang"
#include "Scene/SDFs/SDF3DPrimitiveCommon.slang"
#else
__exported import Utils.Math.AABB;
__exported import Scene.HitInfoType;
__exported import Scene.SDFs.SDF3DPrimitive;
#endif

BEGIN_NAMESPACE_FALCOR

struct SDFGridPlane
{
    float3 position = float3(0.0f, 0.0f, 0.0f);
    float  gridLineWidth = 0.04f;
    float3 normal = float3(0.0f, 0.0f, 1.0f);
    float  gridScale = 25.0f;
    float3 rightVector = float3(1.0f, 0.0f, 0.0f);
    float  planeSize = 0.25f;
    float4 color = float4(0.0f, 0.5f, 1.0f, 0.5f);
    uint   active = 0;                                  // Only render and manipulate the grid if this is true.
    uint3  _pad0;

    float intersect(float3 rayOrigin, float3 rayDir)    // Returns the distance from the ray to the plane.
    {
        return -dot(normal, rayOrigin - position) / dot(normal, rayDir);
    }
};

enum class SDFBBRenderMode : uint32_t
{
    Disabled = 0,
    RenderAll = 1,
    RenderSelectedOnly = 2,

    Count
};

enum class SDFEditorAxis : uint32_t
{
    X = 0,
    Y = 1,
    Z = 2,
    OpSmoothing = 3,
    All = 4,

    Count
};

struct SDFBBRenderSettings
{
    uint renderMode = 2;
    uint selectedInstanceID = 0;
    float edgeThickness = 0.0001f;
};

struct SDFPickingInfo
{
    float distance;
    uint instanceID;
    HitType hitType;
};

struct SDFEditingData
{
    uint editing;
    uint previewEnabled;
    uint instanceID;
    uint scalingAxis;
    SDF3DPrimitive primitive;
    SDF3DPrimitive symmetryPrimitive;
    AABB primitiveBB;
    AABB symmetryPrimitiveBB;
};

END_NAMESPACE_FALCOR
