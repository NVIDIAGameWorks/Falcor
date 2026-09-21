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
import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.IsotropicGGX;
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDHelpers;
import Scene.Scene;
import Scene.ShadingData;
import Utils.Math.MathHelpers;
import PathState;

/** Generate an orthonormal basis that tries to align tangent with the view vector.
    \param[in] v Unit view vector.
    \param[in] n Unit normal vector.
    \param[out] t Unit tangent vector.
    \param[out] b Unit bitangent vector.
*/
void buildOrthonormalBasis(float3 v, float3 n, out float3 t, out float3 b)
{
    static const float kEpsilon = 1.e-6f;

    t = abs(1.f + dot(n, v)) < kEpsilon ? float3(n.z, -n.x, n.y) : v;
    t -= n * dot(n, t);
    t = normalize(t);
    b = cross(n, t);
}

/** Wrap the xform so it operates in local coordinate system aligned with the view vector and normal.
    \param[in] p World position of a point on a plane.
    \param[in] v Unit view vector.
    \param[in] n Unit normal vector.
    \param[in] L Local-space xform.
    \return T Wrapped world-space xform.
*/
float4x4 wrapInLocalXform(float3 p, float3 v, float3 n, float4x4 L)
{
    float3 t1, t2;
    buildOrthonormalBasis(v, n, t1, t2);

    float3x3 R = float3x3(
        t1.x, t2.x, n.x,
        t1.y, t2.y, n.y,
        t1.z, t2.z, n.z
    );

    // LocalToWorld xform.
    float4x4 P = float4x4(
        float4(R[0], p.x),
        float4(R[1], p.y),
        float4(R[2], p.z),
        float4(0.f, 0.f, 0.f, 1.f)
    );

    float3x3 invR = transpose(R);
    float3 t = -mul(invR, p);
    // WorldToLocal xform.
    float4x4 invP = float4x4(
        float4(invR[0], t.x),
        float4(invR[1], t.y),
        float4(invR[2], t.z),
        float4(0.f, 0.f, 0.f, 1.f)
    );

    float4x4 T = mul(L, invP);
    T = mul(P, T);
    return T;
}

/** Generate a reflection xform assuming a flat (zero curvature) reflector.
    \param[in] p World position of a point on the reflecting plane.
    \param[in] v Unit view vector.
    \param[in] n Unit normal vector.
    \param[in] originOffset Offset applied to the ray origin after reflection.
    \return Reflection xform in world space.
*/
float4x4 getReflectionXform(float3 p, float3 v, float3 n, float3 originOffset)
{
    // Planar reflection xform (mirror along local z).
    float4x4 L = float4x4(
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, -1.f, 0.f,
        0.f, 0.f, 0.f, 1.f
    );

    float4x4 originOffsetMat = float4x4(
        1.f, 0.f, 0.f, originOffset.x,
        0.f, 1.f, 0.f, originOffset.y,
        0.f, 0.f, 1.f, originOffset.z,
        0.f, 0.f, 0.f, 1.f
    );

    return mul(originOffsetMat, wrapInLocalXform(p, v, n, L));
}

/** Generate a refraction xform assuming a flat (zero curvature) refractor.
    \param[in] p World position of a point on the refracting plane.
    \param[in] v Unit view vector.
    \param[in] n Unit normal vector (assuming that dot(v, n) >= 0).
    \param[in] ior Relative index of refraction (transmissive IoR / incident IoR).
    \param[in] originOffset Offset applied to the ray origin after refraction.
    \return Refraction xform in world space.
*/
float4x4 getRefractionXform(float3 p, float3 v, float3 n, float ior, float3 originOffset)
{
    const float cos_i = dot(v, n);
    // sint = sini / ior
    // 1 - cost^2 = (1 - cosi^2) / ior^2
    // 1 - (1 - cosi^2) / ior^2 = cost^2
    const float tmp = mad(mad(cos_i, -cos_i, 1.f), -1.f / (ior * ior), 1.f);
    const float cos_t = sqrt(max(0.f, tmp));

    const float l1 = cos_t == 0.f ? 1.f : cos_i;
    const float l2 = cos_t == 0.f ? -1.f : (cos_t * ior);

    // Planar refraction xform.
    float4x4 L = float4x4(
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, l1 / l2, 0.f,
        0.f, 0.f, 0.f, 1.f
    );

    float4x4 originOffsetMat = float4x4(
        1.f, 0.f, 0.f, originOffset.x,
        0.f, 1.f, 0.f, originOffset.y,
        0.f, 0.f, 1.f, originOffset.z,
        0.f, 0.f, 0.f, 1.f
    );

    return mul(originOffsetMat, wrapInLocalXform(p, v, n, L));
}

void setNRDPrimaryHitEmission(
    NRDBuffers outputNRD,
    const bool useNRDDemodulation,
    const PathState path,
    const uint2 pixel,
    const bool isPrimaryHit,
    const float3 emission
)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        // Generate primary hit guide buffers.
        if (useNRDDemodulation)
        {
            outputNRD.primaryHitEmission[pixel] = float4(emission, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            outputNRD.primaryHitEmission[pixel] = 0.f;
        }
    }
}

void setNRDPrimaryHitReflectance(
    NRDBuffers outputNRD,
    const bool useNRDDemodulation,
    const PathState path,
    const uint2 pixel,
    const bool isPrimaryHit,
    const ShadingData sd,
    const BSDFProperties bsdfProperties
)
{
    if (isPrimaryHit && path.getSampleIdx() == 0)
    {
        // Generate primary hit guide buffers.
        if (useNRDDemodulation)
        {
            // Use sum of reflection/transmission albedo as they are denoised together.
            const float3 diffuseReflectance =
                max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
            outputNRD.primaryHitDiffuseReflectance[pixel] = float4(diffuseReflectance, 1.f);

            // We use an approximate shading model with a single specular lobe.
            const float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
            const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
            float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
            specularReflectance = max(kNRDMinReflectance, specularReflectance);
            outputNRD.primaryHitSpecularReflectance[pixel] = float4(specularReflectance, 1.f);
        }
        else
        {
            // Clear buffers on primary hit only if demodulation is disabled.
            outputNRD.primaryHitDiffuseReflectance[pixel] = 1.f;
            outputNRD.primaryHitSpecularReflectance[pixel] = 1.f;
        }
    }
}

void setNRDSampleHitDist(NRDBuffers outputNRD, const PathState path, const uint outSampleIdx)
{
    if (path.getVertexIndex() == 2)
    {
        outputNRD.sampleHitDist[outSampleIdx] = float(path.sceneLength);
    }
}

void setNRDPrimaryHitSeparatedRadiance(
    NRDBuffers outputNRD,
    const bool useNRDDemodulation,
    const PathState path,
    const bool isPrimaryHit,
    const uint outSampleIdx
)
{
    if (isPrimaryHit)
    {
        if (useNRDDemodulation && path.isDeltaReflectionPrimaryHit())
        {
            outputNRD.samplePrimaryHitNEEOnDelta[outSampleIdx] = float4(path.L, 0);
        }
        else
        {
            outputNRD.samplePrimaryHitNEEOnDelta[outSampleIdx] = float4(0.f);
        }
    }
}

void setNRDSampleEmission(
    NRDBuffers outputNRD,
    const bool useNRDDemodulation,
    const PathState path,
    const uint outSampleIdx,
    const bool isPrimaryHit,
    const float3 emission,
    const bool wasDeltaOnlyPathBeforeScattering
)
{
    if (useNRDDemodulation)
    {
        // Always demodulate emission on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            outputNRD.sampleEmission[outSampleIdx] = float4(emission, 1.f);
        }
        // Additionally, accumulate emission along the delta path.
        else if ((path.isDeltaReflectionPrimaryHit() || path.isDeltaTransmissionPath()) && any(emission > 0.f))
        {
            const bool demodulateDeltaReflectionEmission = path.isDeltaReflectionPrimaryHit() && wasDeltaOnlyPathBeforeScattering;
            const bool demodulateDeltaTransmissionEmission = path.isDeltaTransmissionPath() && wasDeltaOnlyPathBeforeScattering;
            if (demodulateDeltaReflectionEmission || demodulateDeltaTransmissionEmission)
            {
                float3 prevEmission = outputNRD.sampleEmission[outSampleIdx].rgb;
                outputNRD.sampleEmission[outSampleIdx] = float4(prevEmission + emission, 1.f);
            }
        }
    }
    else if (isPrimaryHit)
    {
        outputNRD.sampleEmission[outSampleIdx] = 0.f;
    }
}

void setNRDSampleReflectance(
    NRDBuffers outputNRD,
    const bool useNRDDemodulation,
    const PathState path,
    const uint outSampleIdx,
    const bool isPrimaryHit,
    const ShadingData sd,
    const BSDFProperties bsdfProperties,
    const uint lobeTypes,
    const bool wasDeltaOnlyPathBeforeScattering
)
{
    // Demodulate reflectance.
    if (useNRDDemodulation)
    {
        const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        // Always demodulate reflectance from diffuse and specular paths on the primary hit (it seconds as a clear).
        if (isPrimaryHit)
        {
            if (path.isDiffusePrimaryHit())
            {
                // Use sum of reflection/transmission albedo as they are denoised together.
                const float3 diffuseReflectance =
                    max(kNRDMinReflectance, bsdfProperties.diffuseReflectionAlbedo + bsdfProperties.diffuseTransmissionAlbedo);
                outputNRD.sampleReflectance[outSampleIdx] = float4(diffuseReflectance, 1.f);
            }
            else if (path.isSpecularPrimaryHit())
            {
                // We use an approximate shading model with a single specular lobe.
                const float NdotV = saturate(dot(bsdfProperties.guideNormal, sd.V));
                const float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
                float3 specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectionAlbedo, ggxAlpha, NdotV);
                specularReflectance = max(kNRDMinReflectance, specularReflectance);
                outputNRD.sampleReflectance[outSampleIdx] = float4(specularReflectance, 1.f);
            }
            else
            {
                outputNRD.sampleReflectance[outSampleIdx] = 1.f;
            }
        }
        // Demodulate reflectance from the second vertex along delta reflection paths.
        else if (path.isDeltaReflectionPrimaryHit() && wasDeltaOnlyPathBeforeScattering && hasNonDeltaLobes)
        {
            const MaterialType materialType = sd.mtl.getMaterialType();
            const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;
            float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);

            outputNRD.sampleReflectance[outSampleIdx] = float4(deltaReflectance, 1.f);
        }
        // Demodulate reflectance from the first non-delta vertex along delta transmission paths.
        else if (path.isDeltaTransmissionPath() && wasDeltaOnlyPathBeforeScattering && hasNonDeltaLobes)
        {
            const MaterialType materialType = sd.mtl.getMaterialType();
            const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;
            float3 deltaReflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);

            outputNRD.sampleReflectance[outSampleIdx] = float4(deltaReflectance, 1.f);
        }
    }
    else if (isPrimaryHit)
    {
        outputNRD.sampleReflectance[outSampleIdx] = 1.f;
    }
}

/** Write out delta reflection guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaReflectionGuideBuffers(
    NRDBuffers outputNRD,
    const bool useNRDDemodulation,
    const uint2 pixelPos,
    float3 reflectance,
    float3 emission,
    float3 normal,
    float roughness,
    float pathLength,
    float hitDist
)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    outputNRD.deltaReflectionEmission[pixelPos] = float4(emission, 0.f);
    outputNRD.deltaReflectionReflectance[pixelPos] = float4(reflectance, 0.f);
    outputNRD.deltaReflectionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    outputNRD.deltaReflectionPathLength[pixelPos] = pathLength;
    outputNRD.deltaReflectionHitDist[pixelPos] = hitDist;
}

/** Write out delta transmission guide buffers.
    Executed only for guide paths.
*/
void writeNRDDeltaTransmissionGuideBuffers(
    NRDBuffers outputNRD,
    const bool useNRDDemodulation,
    const uint2 pixelPos,
    float3 reflectance,
    float3 emission,
    float3 normal,
    float roughness,
    float pathLength,
    float3 posW
)
{
    emission = useNRDDemodulation ? emission : 0.f;
    reflectance = useNRDDemodulation ? max(kNRDMinReflectance, reflectance) : 1.f;

    float2 octNormal = ndir_to_oct_unorm(normal);
    // Clamp roughness so it's representable of what is actually used in the renderer.
    float clampedRoughness = roughness < 0.08f ? 0.00f : roughness;
    float materialID = 0.f;

    outputNRD.deltaTransmissionEmission[pixelPos] = float4(emission, 0.f);
    outputNRD.deltaTransmissionReflectance[pixelPos] = float4(reflectance, 0.f);
    outputNRD.deltaTransmissionNormWRoughMaterialID[pixelPos] = float4(octNormal, clampedRoughness, materialID);
    outputNRD.deltaTransmissionPathLength[pixelPos] = pathLength;
    outputNRD.deltaTransmissionPosW[pixelPos] = float4(posW, 0.f);
}

float2 calcImageMotionVector(const PathState path, const uint2 frameDim, const HitInfo primaryHit)
{
    // Transform the last vertex's previous-frame world position into the virtual-image space of the
    // primary hit using the accumulated reflection/refraction transforms along the delta path.
    float3 prevPosW = gScene.getPrevPosW(path.hit);
    float4 prevImagePosW = mul(path.imageXform, float4(prevPosW, 1.f));
    prevImagePosW /= prevImagePosW.w;

    // Do the same for the current position.
    float3 currPosW = path.origin;
    float4 currImagePosW = mul(path.imageXform, float4(currPosW, 1.f));
    currImagePosW /= currImagePosW.w;

    float4 currPosH = mul(gScene.camera.data.viewProjMatNoJitter, float4(currImagePosW.xyz, 1.f));
    float2 currCrd = currPosH.xy / currPosH.w;
#ifdef FALCOR_FLIP_Y
    currCrd *= float2(0.5, 0.5);
#else
    currCrd *= float2(0.5, -0.5);
#endif
    currCrd += 0.5f;

    float2 pixelPos = currCrd * frameDim;

    // Compute motion vector in screen space.
    float4 prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevImagePosW.xyz, 1.f));
    // The motion vector already contains no jitter with the current computation.
    float2 imageMotionVector = calcMotionVector(pixelPos, prevPosH, frameDim);

    return imageMotionVector;
}

