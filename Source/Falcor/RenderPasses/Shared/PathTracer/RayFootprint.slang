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
#include "RenderPasses/Shared/PathTracer/RayFootprintModes.slangh"
import Scene.Scene;
import Scene.ShadingData;
import Scene.Shading;
import Scene.Material.MaterialData;
import Scene.HitInfo;
import Utils.Attributes;
import RenderPasses.Shared.PathTracer.PathData;
import RenderPasses.Shared.PathTracer.StaticParams;
__exported import RenderPasses.Shared.PathTracer.RayFootprintData;

#ifndef RAY_FOOTPRINT_MODE
// RAY_FOOTPRINT_MODE must be defined in order to use this module.
#error "RAY_FOOTPRINT_MODE not defined !"
#endif
#ifndef RAY_CONE_MODE
// RAY_CONE_MODE must be defined in order to use this module.
#error "RAY_CONE_MODE not defined !"
#endif
#ifndef RAY_FOOTPRINT_USE_MATERIAL_ROUGHNESS
// RAY_FOOTPRINT_USE_MATERIAL_ROUGHNESS must be defined in order to use this module.
#error "RAY_FOOTPRINT_USE_MATERIAL_ROUGHNESS not defined !"
#endif
#if RAY_CONE_MODE == 0
#error "RayTracingGems1 RAY_CONE_MODE is not supported inside Path Tracers !"
#endif

// Various curvature estimation modes are available. The one selected here seems to be the best compromize, but more experiments are needed.
#define RAY_FOOTPRINT_CURVATURE_ESTIMATOR_AVG(rayDir, rayConeWidth, surfaceSpreadAngle)         TriangleCurvature_Average tce = { }
#define RAY_FOOTPRINT_CURVATURE_ESTIMATOR_MAX(rayDir, rayConeWidth, surfaceSpreadAngle)         TriangleCurvature_Max tce = { }
#define RAY_FOOTPRINT_CURVATURE_ESTIMATOR_DIR(rayDir, rayConeWidth, surfaceSpreadAngle)         TriangleCurvature_Directional tce = { rayDir }
#define RAY_FOOTPRINT_CURVATURE_ESTIMATOR_DIRDP(rayDir, rayConeWidth, surfaceSpreadAngle)       TriangleCurvature_DirClosestDP tce = { rayDir }
#define RAY_FOOTPRINT_CURVATURE_ESTIMATOR_ELLIPSEVIS(rayDir, rayConeWidth, surfaceSpreadAngle)  TriangleCurvature_EllipseVis tce = { rayDir, rayConeWidth, surfaceSpreadAngle }

// Curvature estimators used for primary and secondary hits.
#define RAY_FOOTPRINT_CURVATURE_ESTIMATOR_PRIMARY(rayDir, rayConeWidth, surfaceSpreadAngle)     RAY_FOOTPRINT_CURVATURE_ESTIMATOR_MAX(rayDir, rayConeWidth, surfaceSpreadAngle)
#define RAY_FOOTPRINT_CURVATURE_ESTIMATOR_SECONDARY(rayDir, rayConeWidth, surfaceSpreadAngle)   RAY_FOOTPRINT_CURVATURE_ESTIMATOR_MAX(rayDir, rayConeWidth, surfaceSpreadAngle)

static const TexLODMode kRayFootprintMode = TexLODMode(RAY_FOOTPRINT_MODE);
static const RayConeMode kRayConeMode = RayConeMode(RAY_CONE_MODE);
static const bool kRayFootprintUseMaterialRoughness = RAY_FOOTPRINT_USE_MATERIAL_ROUGHNESS;

// Enable/disable curvature computation to determine the spreading angle at indirect bounces.
static const bool kRayFootprintUseIndirectCurvature = true;
// Enable/disable approximate computation of spreading angle due to surface roughness.
static const bool kRayFootprintUseApproximateRoughnessAngle = false;


float roughnessToVariance(float linearRoughness)
{
    float alpha = linearRoughness * linearRoughness;
    float s = alpha * alpha;
    s = min(s, 0.99);               // Prevents division by zero.
    return (s / (1.0f - s)) * 0.5;
}

float getAngleFromGGXRoughness(float roughness)
{
    if (kRayFootprintUseApproximateRoughnessAngle)
    {
        // Very "rough" approximation.
        return roughness * roughness * 0.25;
    }
    else
    {
        // Correct estimation.
        float sigma2 = roughnessToVariance(roughness);
        return sqrt(sigma2);
    }
}

/** Ray footprint for tex LOD.
    Ray Cone version.
*/
#if RAY_FOOTPRINT_MODE == RAY_FOOTPRINT_MODE_RAYCONE     // Ray Cone.
extension RayFootprint
{
    static RayFootprint create(HitInfo hit, uint2 pixel, uint2 frameDim, float3 worldPos, float3 cameraPos, float3 faceNormal, float3 normal, float screenSpacePixelSpreadAngle, float roughness, bool isSpecular)
    {
        RayFootprint res;

        // Set up ray cone (texLOD) at the hit point: first create ray cone at camera, then propagate to hit point at the end.
        res.rayCone = RayCone.create(0.0, screenSpacePixelSpreadAngle);

        const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(pixel, frameDim);
        const float3 rayDir = normalize(nonNormalizedRayDir);

        const float hitT = length(cameraPos - worldPos.xyz);
        float rayConeWidth = hitT * screenSpacePixelSpreadAngle;

        float surfaceSpreadAngle;
        if (kRayConeMode == RayConeMode::Combo)
        {
            const float4x4 worldMat = gScene.getWorldMatrix(hit.meshInstanceID);
            const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(hit.meshInstanceID);
            const uint3 vertexIndices = gScene.getIndices(hit.meshInstanceID, hit.primitiveIndex);
            const float3 barycentrics = hit.getBarycentricWeights();
            float3 unnormalizedN, normals[3], dNdx, dNdy, edge1, edge2;
            float2 txcoords[3], dBarydx, dBarydy, dUVdx, dUVdy;

            StaticVertexData vertices[3] = { gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2]) };

            RayDiff rayDiff;
            float3 dDdx, dDdy;
            prepareVerticesForRayDiffs(rayDir, vertices, worldMat, worldInvTransposeMat, barycentrics, edge1, edge2, normals, unnormalizedN, txcoords);
            computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, frameDim, dDdx, dDdy);
            RayDiff rd = RayDiff.create(float3(0.0), float3(0.0), dDdx, dDdy);  // Init ray diff. dOdx = 0 , dOdy = 0, and the directions are from above.
            rayDiff = rd.propagate(worldPos.xyz, rayDir, hitT, faceNormal);     // Propagate the ray differential to the current hit point.
            computeBarycentricDifferentials(rayDiff, rayDir, edge1, edge2, faceNormal, dBarydx, dBarydy);
            computeNormalDifferentials(rayDiff, unnormalizedN, dBarydx, dBarydy, normals, dNdx, dNdy);
            float3 right = normalize(gScene.camera.computeNonNormalizedRayDirPinhole(pixel + uint2(1, 0), frameDim)) - rayDir;
            float3 up = normalize(gScene.camera.computeNonNormalizedRayDirPinhole(pixel + uint2(0, 1), frameDim)) - rayDir;
            surfaceSpreadAngle = computeScreenSpaceSurfaceSpreadAngle(right, up, dNdx, dNdy);
        }
        else if (kRayConeMode == RayConeMode::Unified)
        {
            // Various curvature estimation modes are available. cf. RAY_FOOTPRINT_CURVATURE_ESTIMATOR_PRIMARY define at the top of the file.
            RAY_FOOTPRINT_CURVATURE_ESTIMATOR_PRIMARY(rayDir, rayConeWidth, screenSpacePixelSpreadAngle);      // Declares tce, the triangle curvature estimator to use.
            float curvature = gScene.computeCurvatureGeneric(hit.meshInstanceID, hit.primitiveIndex, tce);
            surfaceSpreadAngle = computeSpreadAngleFromCurvatureIso(curvature, rayConeWidth, rayDir, normal);
        }

        if (kRayFootprintUseMaterialRoughness)
        {
            float linearRoughnessForFootprint = roughness;

            // Diffuse path.
            if (!isSpecular)
            {
                // Enable caustic events. TODO: Find a compromize between caustic and LOD speedups.
                linearRoughnessForFootprint = kDisableCaustics ? 0.6 : 0.3;
            }

            // Integrating material roughness.
            surfaceSpreadAngle += getAngleFromGGXRoughness(linearRoughnessForFootprint) * 0.125;       // Primary hits roughness spread change is reduced to 12.5% to avoid overbluring during stochastic sampling.
        }

        // Propagate cone to primary hit point, and reflect over the surface with surfaceSpreadAngle change.
        res.rayCone = res.rayCone.propagate(surfaceSpreadAngle, hitT);

        return res;
    }

    [mutating]
    void hitSurface(inout VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float3 rayOrg, float3 rayDir)
    {
        // Adds Ray Cone data to previously fetched VertexData.
        v.coneTexLODValue = computeRayConeTriangleLODValue(triangleVertices, hit.meshInstanceID, float3x3(gScene.getWorldMatrix(hit.meshInstanceID)));

        float hitT = length(v.posW - rayOrg);
        // Propagate the cone to current surface.
        rayCone = rayCone.propagate(0.0, hitT);
    }

    ShadingData computeShadingData(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float3 rayOrg, float3 rayDir)
    {
        // Compute texture LOD for prepareShadingData().
        float lambda = rayCone.computeLOD(v.coneTexLODValue, rayDir, v.normalW);

        const uint materialID = gScene.getMaterialID(hit.meshInstanceID);
        float3 viewDir = -rayDir;
        return prepareShadingDataUsingRayConesLOD(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], viewDir, lambda);
    }

    // isSpecular, isTransmission : of the new scattered ray.
    [mutating]
    void bounceOnSurface(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, ShadingData sd, float3 normal, float3 rayDirIn, float3 rayDirOut, bool isSpecular, bool isTransmission)
    {
        // Check if scattered ray is refracted.
        if (isTransmission || dot(rayDirOut, normal) <= 0.0) return; // Leaves the footprint unchanged until refraction if properly handled.

        float surfaceSpreadAngle = 0.0f;
        if (kRayFootprintUseIndirectCurvature)
        {
            float rayConeWidth = rayCone.getWidth();
            // Declares tce, the triangle curvature estimator to use.
            RAY_FOOTPRINT_CURVATURE_ESTIMATOR_SECONDARY(rayDir, rayConeWidth, rayCone.getSpreadAngle());
            float curvature = gScene.computeCurvatureGeneric(hit.meshInstanceID, hit.primitiveIndex, tce);
            surfaceSpreadAngle = computeSpreadAngleFromCurvatureIso(curvature, rayConeWidth, rayDirIn, normal);
        }

        if (kRayFootprintUseMaterialRoughness)
        {
            float linearRoughnessForFootprint = sd.linearRoughness;

            // Diffuse path.
            if (!isSpecular)
            {
                // Enable caustic events. TODO: Find a compromize between caustic and LOD speedups.
                linearRoughnessForFootprint = kDisableCaustics ? 0.6 : 0.3;
            }

            // Integrating material roughness.
            surfaceSpreadAngle += getAngleFromGGXRoughness(linearRoughnessForFootprint);
        }

        // Reflect the cone: update spread angle.
        rayCone = rayCone.propagate(surfaceSpreadAngle, 0.0);
    }
};

/** Ray footprint for tex LOD.
    Ray Diff version.
*/
#elif ((RAY_FOOTPRINT_MODE == RAY_FOOTPRINT_MODE_RAYDIFFS_ISO) || (RAY_FOOTPRINT_MODE == RAY_FOOTPRINT_MODE_RAYDIFFS_ANISO))

extension RayFootprint
{
    static RayFootprint create(HitInfo hit, uint2 pixel, uint2 frameDim, float3 worldPos, float3 cameraPos, float3 faceNormal, float3 normal, float screenSpacePixelSpreadAngle, float roughness, bool isSpecular)
    {
        RayFootprint res;

        const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(pixel, frameDim);
        const float3 rayDir = normalize(nonNormalizedRayDir);

        float3 dDdx, dDdy;
        const float3 rayOrg = gScene.camera.getPosition();
        const float hitT = length(rayOrg - worldPos.xyz);

        computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, frameDim, dDdx, dDdy);
        res.rayDiff = RayDiff.create(float3(0.0), float3(0.0), dDdx, dDdy);             // init ray diff. dOdx = 0 , dOdy = 0, and the directions are from above.
        res.rayDiff = res.rayDiff.propagate(worldPos.xyz, rayDir, hitT, faceNormal);    // Propagate the ray differential to the current hit point.

        // Bounce using primary hit geometry.
        const float4x4 worldMat = gScene.getWorldMatrix(hit.meshInstanceID);
        const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(hit.meshInstanceID);
        const uint3 vertexIndices = gScene.getIndices(hit.meshInstanceID, hit.primitiveIndex);
        const float3 barycentrics = hit.getBarycentricWeights();
        float3 unnormalizedN, normals[3], dNdx, dNdy, edge1, edge2;
        float2 txcoords[3], dBarydx, dBarydy, dUVdx, dUVdy;

        StaticVertexData vertices[3] = { gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2]) };
        prepareVerticesForRayDiffs(rayDir, vertices, worldMat, worldInvTransposeMat, barycentrics, edge1, edge2, normals, unnormalizedN, txcoords);

        computeBarycentricDifferentials(res.rayDiff, rayDir, edge1, edge2, faceNormal, dBarydx, dBarydy);
        reflectRayDifferential(res.rayDiff, rayDir, unnormalizedN, normalize(unnormalizedN), dBarydx, dBarydy, normals);

        return res;
    }

    [mutating]
    void hitSurface(inout VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float3 rayOrg, float3 rayDir)
    {
        // Propagate to hit point.
        float hitT = length(v.posW - rayOrg);
        float3 geometricNormal = gScene.getFaceNormalW(hit.meshInstanceID, hit.primitiveIndex);
        this.rayDiff = this.rayDiff.propagate(rayOrg, rayDir, hitT, geometricNormal);   // Propagate the ray differential to the current hit point.
    }

    ShadingData computeShadingData(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float3 rayOrg, float3 rayDir)
    {
        float hitT = length(v.posW - rayOrg);

        // Get hit point adapted parameters.
        float2 dUVdx, dUVdy;    // Ray differential variables for the texture lookup.
        prepareRayDiffAtHitPoint(v, triangleVertices, hit.getBarycentricWeights(), rayDir, hitT, gScene.getWorldMatrix(hit.meshInstanceID), gScene.getInverseTransposeWorldMatrix(hit.meshInstanceID),
            this.rayDiff, dUVdx, dUVdy);

        // Compute shading data.
        float3 viewDir = -rayDir;
        ShadingData sd;
        const uint materialID = gScene.getMaterialID(hit.meshInstanceID);
        if (kRayFootprintMode == TexLODMode::RayDiffsIsotropic)
        {
            // When using prepareShadingDataUsingRayDiffsLOD(), the texture sampler will compute a single lambda for texture LOD
            // using the dUVdx and dUVdy.
            sd = prepareShadingDataUsingRayDiffsLOD(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], viewDir, dUVdx, dUVdy);
        }
        else        // TexLODMode::RayDiffsAnisotropic.
        {
            // Use the dUVdx and dUVdy directly --> gives you anisotropic filtering.
            sd = prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], viewDir, dUVdx, dUVdy);
        }

        return sd;
    }

    [mutating]
    void bounceOnSurface(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, ShadingData sd, float3 normal, float3 rayDirIn, float3 rayDirOut, bool isSpecular, bool isTransmission)
    {
        // Check if scattered ray is refracted.
        if (isTransmission || dot(rayDirOut, normal) <= 0.0) return; // Leaves the footprint unchanged until refraction if properly handled.

        // TODO: Avoid recomputing everything (dUVdx, dUVdy and  dBarydx, dBarydy) by keeping intermediate variables live in registers.
        float2 dUVdx, dUVdy;    // Ray differential variables for the texture lookup.
        reflectRayDiffUsingVertexData(v, triangleVertices, hit.getBarycentricWeights(), rayDirIn, gScene.getWorldMatrix(hit.meshInstanceID), gScene.getInverseTransposeWorldMatrix(hit.meshInstanceID),
                                        this.rayDiff, dUVdx, dUVdy);
    }
};

/** Ray footprint for tex LOD.
    Force LOD0 version.
    Makes the interface consistant even when ray footprint is disabled.
*/
#else
extension RayFootprint
{
    static RayFootprint create(HitInfo hit, uint2 pixel, uint2 frameDim, float3 worldPos, float3 cameraPos, float3 faceNormal, float3 normal, float screenSpacePixelSpreadAngle, float roughness, bool isSpecular)
    {
        RayFootprint res;
        return res;
    }

    [mutating]
    void hitSurface(inout VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float3 rayOrg, float3 rayDir)
    {
    }

    [mutating]
    void bounceOnSurface(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, ShadingData sd, float3 normal, float3 rayDirIn, float3 rayDirOut, bool isSpecular, bool isTransmission)
    {
    }

    ShadingData computeShadingData(VertexData v, StaticVertexData triangleVertices[3], HitInfo hit, float3 rayOrg, float3 rayDir)
    {
        const uint materialID = gScene.getMaterialID(hit.meshInstanceID);
        return prepareShadingData(v, materialID, gScene.materials[materialID], gScene.materialResources[materialID], -rayDir, 0.f);
    }
};
#endif

/** Prepare shading data using ray footprint and TexLOD.
   \param[in] v The mesh vertex data interpolated at hit point.
   \param[in] rayFootprint The ray footprint to use for TexLOD.
   \param[in] triangleVertices The vertex data of the 3 vertices of the triangle.
   \param[in] hit Ray hit info at hit point.
   \param[in] rayOrg Ray origin.
   \param[in] rayDir Ray direction.
   \return Shading data computed at hit point.
*/
ShadingData prepareShadingData(VertexData v, RayFootprint rayFootprint, StaticVertexData triangleVertices[3], HitInfo hit, float3 rayOrg, float3 rayDir)
{
    return rayFootprint.computeShadingData(v, triangleVertices, hit, rayOrg, rayDir);
}
