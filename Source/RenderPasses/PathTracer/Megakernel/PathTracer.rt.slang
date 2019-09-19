/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
***************************************************************************/

/** Path tracing pass.

    This file contains the entry points for all ray tracing programs.
    We import the path tracer utility functions defined in PathTracer.slang.

    The host sets the compile-time constants in StaticParams.slang.
    It also sets the following defines for optional I/O buffers:

    is_valid_<name> is 1 if buffer with this name is bound, 0 otherwise.
*/

import PathTracer;
import Utils.Math.MathHelpers;
import RenderPasses.PathTracer.LoadGBuffer;

shared ParameterBlock<PathTracerData> gData;

// Outputs (optional)
shared RWTexture2D<float4> gOutputColor;
shared RWTexture2D<float4> gOutputAlbedo;
shared RWTexture2D<float4> gOutputDirect;
shared RWTexture2D<float4> gOutputIndirect;

// Static configuration based on which buffers are bound.
#define isValid(name) (is_valid_##name != 0)
static const bool kComputeDirect = isValid(gOutputColor) || isValid(gOutputDirect);
static const bool kComputeIndirect = (isValid(gOutputColor) || isValid(gOutputIndirect)) && kMaxBounces > 0;


/** ********************* Ray index 0: Scatter ray ************************ */

[shader("miss")]
void scatterMiss(inout ScatterRayDataPacked packedData : SV_RayPayload)
{
    ScatterRayData rayData = packedData.unpack();
    handleMiss(gData, rayData);
    packedData.pack(rayData);
}

[shader("anyhit")]
void scatterAnyHit(inout ScatterRayDataPacked packedData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(PrimitiveIndex(), attribs);
    if (alphaTest(v, gScene.materials[gScene.getMaterialID(getGlobalHitID())], 0.f)) IgnoreHit();
}

[shader("closesthit")]
void scatterClosestHit(inout ScatterRayDataPacked packedData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    ScatterRayData rayData = packedData.unpack();

    // Evaluate Falcor's material parameters at the hit point.
    // Note we pass hitPos-rayDir as "camera position" to avoid zero-length rays causing NaNs
    // in the view direction. It'd been cleaner if prepareShadingData() took ray dir directly.
    // TODO: Implement texLOD to enable texture filtering in prepareShadingData().
    const float3 rayDir = WorldRayDirection();
    VertexData v = getVertexData(PrimitiveIndex(), attribs);
    ShadingData sd = prepareShadingData(v, gScene.materials[gScene.getMaterialID(getGlobalHitID())], v.posW - rayDir, 0.f);

    // Compute tangent space if it is invalid.
    if (!(dot(sd.B, sd.B) > 0.f))   // Note: Comparison written so that NaNs trigger
    {
        sd.B = perp_stark(sd.N);
        sd.T = cross(sd.B, sd.N);
    }

    // Prepare hit point struct with the additional data not part of ShadingData.
    // These fields are needed for ray offset computatation and light PDF evaluation.
    TriangleHit hit;
    hit.meshInstanceID = getGlobalHitID();
    hit.primitiveIndex = PrimitiveIndex();
    hit.posW = sd.posW;
    hit.normalW = sd.frontFacing ? sd.faceN : -sd.faceN;
    hit.triangleArea = gScene.getFaceAreaW(hit.meshInstanceID, hit.primitiveIndex);

    handleHit(gData, sd, hit, kComputeIndirect, rayData);
    packedData.pack(rayData);
}

/************************** Ray index 1: Shadow ray ************************ */

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData : SV_RayPayload)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(PrimitiveIndex(), attribs);
    if (alphaTest(v, gScene.materials[gScene.getMaterialID(getGlobalHitID())], 0.f)) IgnoreHit();
}

/** ******************************** RayGen ******************************** */

/** This is the entry point for the path tracer.

    We generate N paths (= #spp) per pixel, which are traced into the scene.
    The path tracer is written as a for-loop over path segments, where each
    iteration traces a shadow ray for direct illumination and a scatter ray.

    The hit shader for the scatter ray currently generates ray parameters for
    the shadow ray to evaluate direct illumination and generates ray parameters
    for the next scatter ray, which are both returned the raygen shader to be
    traced. This is more efficient than tracing from the hit shader. The max
    recusion depth = 1.
*/
[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    float3 outDirect = float3(0, 0, 0);
    float3 outIndirect = float3(0, 0, 0);
    float3 outAlbedo = float3(0, 0, 0);
    float outAlpha = 0.f;

    ShadingData sd;
    if (loadShadingData(launchIndex, launchDim, gScene.camera, sd))
    {
        // Pixel represents a primary hit. Compute its contribution.

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin = sd.computeNewRayOrigin();

        // Loop over samples in pixel.
        for (uint sampleIdx = 0; sampleIdx < kSamplesPerPixel; sampleIdx++)
        {
            // Create sample generator.
            uint frameSeed = gData.params.useFixedSeed ? 0 : gData.params.frameCount;
            SampleGenerator sg = SampleGenerator.create(launchIndex, frameSeed * kSamplesPerPixel + sampleIdx);

            // Advance the generator to the first available dimension.
            // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
            for (uint i = 0; i < gData.params.prngDimension; i++) sampleNext1D(sg);

            // Prepare ray payload.
            ScatterRayData rayData = ScatterRayData.create(sg);

            float3 Ldirect = float3(0, 0, 0);
            float3 Lindirect = float3(0, 0, 0);

            // Always output directly emitted light from the primary hit (unclamped as it's noise free).
            // This is independent of whether emissive materials are treated as light sources or not.
            outDirect += sd.emissive;

            // Sample direct illumination at primary hit.
            if (kComputeDirect)
            {
                Ldirect += evalDirect(gData, sd, rayOrigin, rayData.sg);
            }

            // Generate ray parameters for the first path segment.
            generateScatterRay(gData, sd, rayOrigin, rayData);

            // Compute iteration count.
            // This is statically determined based on the current configuration, is MIS/emissive is needed at the last bounce etc.
            const uint kLastBounce = kComputeIndirect ? kMaxBounces : 0;
            const uint kIterationCount = kTraceScatterRayFromLastPathVertex ? kLastBounce + 1 : kLastBounce;

            [unroll]
            for (uint depth = 0; depth < kIterationCount && !rayData.terminated; depth++)
            {
                // Reset contributions.
                rayData.Le = rayData.Lr = float3(0);

                // Trace scatter ray. The closest hit shader generates a shadow ray and a new scatter ray.
                traceScatterRay(rayData);

                // Accumulate emitted radiance as direct/indirect depending on path length.
                if (depth == 0) Ldirect += rayData.Le;
                else Lindirect += rayData.Le;

                if (depth < kLastBounce)
                {
                    // Trace shadow ray and accumulate reflected radiance as indirect if light is visible.
                    bool shadowValid = any(rayData.Lr > 0.f);
                    bool V = traceShadowRay(rayData.origin, rayData.shadowRay.xyz, rayData.shadowRay.w, shadowValid);
                    if (V) Lindirect += rayData.Lr;

                    // Russian roulette to stochastically terminate the path.
                    // We use a fixed absorption probability for now.
                    // TODO: Better strategy, e.g., 1-P(absorption) = hemispherical reflectance of BRDF.
                    if (kUseRussianRoulette)
                    {
                        float u = sampleNext1D(rayData.sg);
                        if (u < gData.params.probabilityAbsorption) rayData.terminated = true;
                        rayData.thp /= (1.f - gData.params.probabilityAbsorption);
                    }
                }
            }
            logPathLength(rayData.pathLength);

            // Accumulate direct/indirect illumination after clamping.
            // Note the comparisons are purposely written so that NaNs propagate (unless the compiler rewrites it).
            outDirect += gData.params.clampDirect && Ldirect > gData.params.thresholdDirect ? gData.params.thresholdDirect : Ldirect;
            outIndirect += gData.params.clampIndirect && Lindirect > gData.params.thresholdIndirect ? gData.params.thresholdIndirect : Lindirect;
        }

        // We're done accumulating over all samples.
        const float invSpp = 1.f / kSamplesPerPixel;
        outDirect *= invSpp;
        outIndirect *= invSpp;
        outAlbedo = sd.diffuse + sd.specular;
        outAlpha = 1.f;
    }
    else
    {
        // Background pixel.
        outDirect = evalBackground(gData.envProbe, -sd.V);
        outIndirect = float3(0, 0, 0);
        outAlpha = kForceAlphaOne ? 1.f : 0.f;
        outAlbedo = outDirect.rgb;
    }

    // Write outputs.
    // These are all optional so using compile-time checks to decide which ones to write.
    float3 outColor = outDirect + outIndirect;
    assert(!any(isnan(outColor)));

    // DEBUG
    //if (any(isnan(outColor))) outColor = float3(1, 0, 0);
    //else if (any(isinf(outColor))) outColor = float3(0, 1, 0);

    if (isValid(gOutputColor)) gOutputColor[launchIndex] = float4(outColor, outAlpha);
    if (isValid(gOutputDirect)) gOutputDirect[launchIndex] = float4(outDirect, outAlpha);
    if (isValid(gOutputIndirect)) gOutputIndirect[launchIndex] = float4(outIndirect, outAlpha);
    if (isValid(gOutputAlbedo)) gOutputAlbedo[launchIndex] = float4(outAlbedo, 1);
}
