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
#include "Utils/Math/MathConstants.slangh"

import Scene.SDFs.SDF3DPrimitive;
import Utils.Math.PackedFormats;

#define GROUP_WIDTH 4

cbuffer CB
{
    uint gGridWidth;
    uint gPrimitiveCount;
};

StructuredBuffer<SDF3DPrimitive> gPrimitives;
Texture3D<float> gOldValues;

#ifdef _USE_SD_FIELD_3D_TEXTURE
RWTexture3D<float> gValues;
#else
RWBuffer<float> gValues;
#endif

void writeDistanceToValues(uint3 valueCoords, float sd)
{
#ifdef _USE_SD_FIELD_3D_TEXTURE
    gValues[valueCoords] = sd;
#else
    uint writeLocation = flatten3D(valueCoords, gGridWidth + 1, gGridWidth + 1);
    gValues[writeLocation] = sd;
#endif
}

[numthreads(GROUP_WIDTH, GROUP_WIDTH, GROUP_WIDTH)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupThreadID : SV_GroupThreadID)
{
    if (any(dispatchThreadID >= gGridWidth + 1)) return;

    const uint3 valueCoords = dispatchThreadID;

    // Calculate the grid position of the value coordinates.
    const float3 p = -0.5f + float3(valueCoords) / gGridWidth;

    // Evaluate the signed distance value.
    float sd = FLT_MAX;

#ifdef _MERGE_WITH_THE_SD_FIELD
    sd = gOldValues[valueCoords];
#endif

    for (uint primitiveID = 0; primitiveID < gPrimitiveCount; primitiveID++)
    {
        SDF3DPrimitive primitive = gPrimitives[primitiveID];
        sd = primitive.eval(p, sd);
    }

    writeDistanceToValues(valueCoords, sd);
}
