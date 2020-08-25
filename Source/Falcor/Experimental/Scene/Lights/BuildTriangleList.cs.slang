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

import Scene.Scene;
import Utils.Color.ColorHelpers;
import Experimental.Scene.Lights.LightCollection;

cbuffer CB
{
    uint gLightIdx;                         ///< The mesh light index.
    uint gMaterialID;                       ///< The material ID of the mesh light.
    uint gMeshInstanceID;                   ///< Mesh light's global mesh instance ID. We use this to find its data in the scene.
    uint gTriangleCount;                    ///< Number of triangles in current mesh light.
    uint gTriangleOffset;                   ///< Current offset into global list of emissive triangles.
}

RWStructuredBuffer<PackedEmissiveTriangle> gTriangleData;   ///< Per-triangle data for emissive triangles.

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

    // Fetch vertex data.
    EmissiveTriangle tri;
    gScene.getVertexPositionsW(gMeshInstanceID, triangleIndex, tri.posW);
    gScene.getVertexTexCoords(gMeshInstanceID, triangleIndex, tri.texCoords);
    tri.normal = gScene.computeFaceNormalAndAreaW(gMeshInstanceID, tri.posW, tri.area);;
    tri.materialID = gMaterialID;
    tri.lightIdx = gLightIdx;

    // Store triangle data.
    gTriangleData[triIdx].pack(tri);
}
