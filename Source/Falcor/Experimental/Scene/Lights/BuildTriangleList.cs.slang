/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Color.ColorHelpers;
import Experimental.Scene.Lights.LightCollection;

cbuffer CB
{
    uint gLightIdx;                         ///< The mesh light index.
    uint gMeshInstanceID;                   ///< Mesh light's global mesh instance ID. We use this to find its data in the scene.
    uint gTriangleCount;                    ///< Number of triangles in current mesh light.
    uint gTriangleOffset;                   ///< Current offset into global list of emissive triangles.
}

RWByteAddressBuffer gVertexPosOutput;       ///< Vertex positions in world space for all mesh light triangles. Size: triangleCount * 3 * sizeof(float3).
RWByteAddressBuffer gTexCoordsOutput;       ///< Texture coordinates for all mesh light triangles. Size: triangleCount * 3 * sizeof(float2).

RWStructuredBuffer<EmissiveTriangle> gTriangleData;     ///< Per-triangle data for emissive triangles. Size: triangleCount * sizeof(EmissiveTriangle).


/** Kernel building the emissive triangles list for all mesh lights.
    One dispatch per mesh light with one thread per triangle.
*/
[numthreads(256, 1, 1)]
void buildTriangleList(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gTriangleCount) return;

    const uint triangleIndex = DTid.x; // Triangle index in current mesh.
    const uint triIdx = gTriangleOffset + triangleIndex;
    const uint vtxIdx = triIdx * 3; // The emissive triangle list is non-indexed.

    // Store vertex data.
    float3 p[3];
    float2 texC[3];
    gScene.getVertexPositionsW(gMeshInstanceID, triangleIndex, p);
    gScene.getVertexTexCoords(gMeshInstanceID, triangleIndex, texC);

    [unroll]
    for (int i = 0; i < 3; i++)
    {
        gVertexPosOutput.Store3((vtxIdx + i) * 12, asuint(p[i]));
        gTexCoordsOutput.Store2((vtxIdx + i) * 8, asuint(texC[i]));
    }

    // Store triangle data.
    float triangleArea;
    float3 faceNormal = gScene.computeFaceNormalAndAreaW(gMeshInstanceID, p, triangleArea);

    gTriangleData[triIdx].lightIdx = gLightIdx;
    gTriangleData[triIdx].normal = faceNormal;
    gTriangleData[triIdx].area = triangleArea;
}
