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
import Scene.Raster;
import Utils.Math.MathHelpers;

RasterizerOrderedTexture2D<float2> gMotionVector;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

#define is_valid(name) (is_valid_##name != 0)

struct VBufferPSOut
{
    PackedHitInfo packedHitInfo : SV_TARGET0;
};

struct VBufferVSOut
{
    float2 texC : TEXCRD;

    // Per-triangle data
    nointerpolation GeometryInstanceID instanceID   : INSTANCE_ID;
    nointerpolation uint materialID                 : MATERIAL_ID;

    float4 posH : SV_POSITION;  ///< Position in world space.

#if is_valid(gMotionVector)
    float4 prevPosH : PREVPOSH; ///< Position in clip space for the previous frame.
#endif
};

VBufferVSOut vsMain(VSIn vsIn)
{
    VBufferVSOut vsOut;
    const GeometryInstanceID instanceID = { vsIn.instanceID };

    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float3 posW = mul(worldMat, float4(vsIn.pos, 1.f)).xyz;
    vsOut.posH = mul(gScene.camera.getViewProj(), float4(posW, 1.f));

    vsOut.texC = vsIn.texC;
    vsOut.instanceID = instanceID;
    vsOut.materialID = gScene.getMaterialID(instanceID);

#if is_valid(gMotionVector)
    // Compute the vertex position in the previous frame.
    float3 prevPos = vsIn.pos;
    GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
    if (instance.isDynamic())
    {
        uint prevVertexIndex = gScene.meshes[instance.geometryID].prevVbOffset + vsIn.vertexID;
        prevPos = gScene.prevVertices[prevVertexIndex].position;
    }
    float3 prevPosW = mul(gScene.getPrevWorldMatrix(instanceID), float4(prevPos, 1.f)).xyz;
    vsOut.prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));
#endif

    return vsOut;
}

VBufferPSOut psMain(VBufferVSOut vsOut, uint triangleIndex : SV_PrimitiveID, float3 barycentrics : SV_Barycentrics)
{
    VBufferPSOut psOut;

    // Using vsOut.posH.xy as pixel coordinate since it has the SV_Position semantic.
    int2 ipos = int2(vsOut.posH.xy);

#if USE_ALPHA_TEST
    // Alpha test.
    // With the default alpha test, only texC of VertexData is needed.
    // Alpha textures is sampled at lod=0 to match the ray traced V-buffer.
    VertexData v = {};
    v.texC = vsOut.texC;
    float lod = 0.f;

    if (gScene.materials.alphaTest(v, vsOut.materialID, lod)) discard;
#endif

    // Note on barycentrics:
    // The barycentric weights provided to pixel shader correspond to vertices A, B, C of the rasterized triangle.
    // For triangle strips, every odd primitive has the order for vertices B and C flipped. We don't handle triangle
    // strips as DXR does not support them and Falcor uses triangle lists exclusively in its scene representation.
    // DXR intersection attributes store barycentric weights in a float2 for triangle vertices B and C.
    // This is what we store in the hit info.

    // Store hit information.
    TriangleHit triangleHit;
    triangleHit.instanceID = vsOut.instanceID;
    triangleHit.primitiveIndex = triangleIndex;
    triangleHit.barycentrics = barycentrics.yz;
    psOut.packedHitInfo = triangleHit.pack();

#if is_valid(gMotionVector)
    // Compute motion vector.
    {
        const float2 pixelPos = ipos + float2(0.5, 0.5); // Current sample in pixel coords.
        const float4 prevPosH = vsOut.prevPosH; // Sample in previous frame in clip space coords, no jittering applied.
        const float2 mv = calcMotionVector(pixelPos, prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY); // Remove camera jitter from motion vector
        gMotionVector[ipos] = mv;
    }
#endif

    return psOut;
}
