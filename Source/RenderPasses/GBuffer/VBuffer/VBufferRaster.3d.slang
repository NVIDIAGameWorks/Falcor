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
import Scene.Raster;
import Scene.HitInfo;

struct VBufferPSOut
{
    uint2 packedHitInfo : SV_TARGET0;
};

struct VBufferVSOut
{
    float2 texC : TEXCRD;

    // Per-triangle data
    nointerpolation uint meshInstanceID : DRAW_ID;
    nointerpolation uint materialID : MATERIAL_ID;

    float4 posH : SV_POSITION;
};

VBufferVSOut vsMain(VSIn vsIn)
{
    VBufferVSOut vsOut;

    float4x4 worldMat = gScene.getWorldMatrix(vsIn.meshInstanceID);
    float4 posW = mul(float4(vsIn.pos, 1.f), worldMat);
    vsOut.posH = mul(posW, gScene.camera.getViewProj());

    vsOut.texC = vsIn.texC;
    vsOut.meshInstanceID = vsIn.meshInstanceID;
    vsOut.materialID = gScene.getMaterialID(vsIn.meshInstanceID);

    return vsOut;
}

VBufferPSOut psMain(VBufferVSOut vsOut, uint triangleIndex : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics)
{
    VBufferPSOut psOut;

#if !DISABLE_ALPHA_TEST
    // Alpha test.
    // With the default alpha test, only texC of VertexData is needed.
    // Alpha textures is sampled at lod=0 to match the ray traced V-buffer.
    VertexData v = {};
    v.texC = vsOut.texC;
    float lod = 0.f;

    if (alphaTest(v, gScene.materials[vsOut.materialID], gScene.materialResources[vsOut.materialID], lod)) discard;
#endif

    // Note on barycentrics:
    // The barycentric weights provided to pixel shader correspond to vertices A, B, C of the rasterized triangle.
    // For triangle strips, every odd primitive has the order for vertices B and C flipped. We don't handle triangle
    // strips as DXR does not support them and Falcor uses triangle lists exclusively in its scene representation.
    // DXR intersection attributes store barycentric weights in a float2 for triangle vertices B and C.
    // This is what we store in the hit info.

    // Store hit information.
    HitInfo hit;
    hit.meshInstanceID = vsOut.meshInstanceID;
    hit.primitiveIndex = triangleIndex;
    hit.barycentrics = barycentrics.yz;
    psOut.packedHitInfo = hit.encode();

    return psOut;
}
