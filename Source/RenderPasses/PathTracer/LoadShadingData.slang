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
#include "Scene/SceneDefines.slangh"
__exported import Scene.Shading;
import Scene.Material.ShadingUtils;
import Rendering.Materials.TexLODHelpers;
import Utils.Math.MathHelpers;
import Params;

/** Compute derivatives at primary triangle hit.
*/
void computeDerivativesAtPrimaryTriangleHit(const TriangleHit hit, uint2 pixel, uint2 frameDim, out float2 ddx, out float2 ddy)
{
    // Get world space position
    VertexData v = gScene.getVertexData(hit);
    float hitT = length(v.posW - gScene.camera.getPosition());
    const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(pixel, frameDim);
    float3 rayDir = normalize(nonNormalizedRayDir);

    float3 dDdx, dDdy;
    computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, frameDim, dDdx, dDdy);

    RayDiff rd = RayDiff(float3(0.f), float3(0.f), dDdx, dDdy);
    rd = rd.propagate(v.posW.xyz, rayDir, hitT, v.faceNormalW);

    // Bounce using primary hit geometry.
    const float4x4 worldMat = gScene.getWorldMatrix(hit.instanceID);
    const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(hit.instanceID);
    const uint3 vertexIndices = gScene.getIndices(hit.instanceID, hit.primitiveIndex);
    StaticVertexData vertices[3] = { gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2]) };
    float2 dBarydx, dBarydy;
    float3 unnormalizedN, normals[3];

    prepareRayDiffAtHitPoint(v, vertices, hit.getBarycentricWeights(), rayDir, worldMat, worldInvTransposeMat, rd, unnormalizedN, normals, dBarydx, dBarydy, ddx, ddy);
}

/** Setup vertex data based on hit point.
    \param[in] hit Hit information.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Normalized ray direction.
    \param[out] materialID Material ID.
    \return VertexData struct.
*/
VertexData loadVertexData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, out uint materialID, out bool modifyNormals)
{
    VertexData v = {};
    materialID = {};
    modifyNormals = true;

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        StaticVertexData vertices[3];
        v = gScene.getVertexData(triangleHit, vertices);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif
    return v;
}

/** Adjust ShadingData
    \param[in,out] sd Shading data to update.
    \param[in] hit Hit information.
    \param[in] v Vertex data.
    \param[in] isPrimary True if this is the primary hit point.
*/
void adjustShadingData(inout ShadingData sd, const HitInfo hit, const VertexData v, const bool isPrimary)
{
    if (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle)
    {
#if GBUFFER_ADJUST_SHADING_NORMALS
        if (kAdjustShadingNormals || isPrimary) adjustShadingNormal(sd, v);
#else
        if (kAdjustShadingNormals && !isPrimary) adjustShadingNormal(sd, v);
#endif
    }
}

/** Setup ShadingData based on loaded vertex/material attributes for a hit point.
    \param[in] hit Hit information.
    \param[in] rayOrigin Ray origin.
    \param[in] rayDir Normalized ray direction.
    \param[in] isPrimary True if this is the primary hit point.
    \param[in] lod Method for computing texture level-of-detail
    \return ShadingData struct.
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const bool isPrimary, const ITextureSampler lod)
{
    uint materialID = {};
    bool modifyNormals = true;

    VertexData v = loadVertexData(hit, rayOrigin, rayDir, materialID, modifyNormals);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod, modifyNormals);
    adjustShadingData(sd, hit, v, isPrimary);
    return sd;
}
