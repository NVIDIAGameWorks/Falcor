/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDHelpers;
import LoadShadingData;
import PathTracer;
import NRDHelpers;

// Screen-space motion vector buffers owned by ReSTIR PT (`mpDeltaReflectionMotionVectors` /
// `mpDeltaTransmissionMotionVectors`). Bound at the trace pass root-var so they can be written
// independently of the NRD guide buffers;
RWTexture2D<float2> deltaReflectionMotionVectors;
RWTexture2D<float2> deltaTransmissionMotionVectors;

/** Extends the PathTracer with variants of handleHit() to trace NRD guide paths.
 */
extension PathTracer
{
    /** Handle hit on delta reflection materials.
        After handling the hit, the path is terminated.
        Executed only for guide paths.
        \param[in,out] path The path state.
    */
    void handleDeltaReflectionHit(inout PathState path)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Write out reflectance/normalWRough/posW of the second path vertex
        // - Terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir);

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path))
        {
            float3 originOffset = sd.posW - path.origin;

            float4x4 originOffsetMat = float4x4(
                1.f, 0.f, 0.f, originOffset.x,
                0.f, 1.f, 0.f, originOffset.y,
                0.f, 0.f, 1.f, originOffset.z,
                0.f, 0.f, 0.f, 1.f
            );

            path.imageXform = mul(path.imageXform, originOffsetMat);

            return;
        }

        // Create texture sampler.
        let lod = createTextureSampler(path, isPrimaryHit);

        // Create material instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);
#if USE_RESTIRPT // allow computing specmvec for non-delta surface for ReSTIR PT tempora reuse
        bool isSmooth = bsdfProperties.roughness <= restirpt.specularMotionVectorRoughnessThreshold;
#else
        bool isSmooth = bsdfProperties.roughness == 0.f;
#endif        
        uint smoothMask = uint(isSmooth) * uint(LobeType::All);

        // Query BSDF lobes.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;

        const MaterialType materialType = sd.mtl.getMaterialType();

        if (isPrimaryHit)
        {
            // Terminate without the write-out if the path doesn't start as delta reflection.
            bool hasDeltaReflectionLobe = ((lobeTypes & (uint)LobeType::DeltaReflection) != 0) || ((lobeTypes & (uint)LobeType::SpecularReflection) != 0) && isSmooth;
            if (!hasDeltaReflectionLobe)
            {
                if (kOutputNRDAdditionalData)
                {
                    writeNRDDeltaReflectionGuideBuffers(
                        outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength
                    );
                }
                if (kOutputDeltaMotionVectors)
                {
                    deltaReflectionMotionVectors[pixel] = 0.f;
                }
                path.terminate();
                return;
            }

            // Add primary ray length to the path length.
            float primaryHitDist = length(sd.posW - path.origin);
            path.sceneLength += float16_t(primaryHitDist);
            // Hijack pdf that we don't need.
            path.pdf += primaryHitDist;

            // Set the active lobes only to delta reflection on the first bounce.
            sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection | smoothMask & (uint)LobeType::SpecularReflection);
        }
        else
        {
            // disable soft roughness for delta for ReSTIR specmvec because it conflicts with NRD denoiser output
            if (kOutputNRDData && path.isDeltaReflectionPrimaryHit()) isSmooth = bsdfProperties.roughness == 0.f;

            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * bsdfProperties.emission;

            // Terminate after scatter ray on last vertex has been processed or non-delta lobe exists.
            const bool lastVertex = hasFinishedSurfaceBounces(path);
            const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0 && !isSmooth || (lobeTypes & (uint)LobeType::Diffuse);
            const bool isEmissive = any(bsdfProperties.emission > 0.f);

            if (lastVertex || hasNonDeltaLobes || isEmissive)
            {
                const float3 emission = path.L;
                const float3 reflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);
                const float primaryHitDist = path.pdf;
                const float hitDist = float(path.sceneLength) - primaryHitDist;
                if (kOutputNRDAdditionalData)
                {
                    writeNRDDeltaReflectionGuideBuffers(
                        outputNRD,
                        kUseNRDDemodulation,
                        pixel,
                        reflectance,
                        emission,
                        bsdfProperties.guideNormal,
                        bsdfProperties.roughness,
                        float(path.sceneLength),
                        hitDist
                    );
                }
                if (kOutputDeltaMotionVectors)
                {
                    const HitInfo primaryHit = unpackHitInfo(vbuffer[pixel]);
                    // Place the current path origin at the terminal hit so calcImageMotionVector can project it.
                    path.origin = sd.posW;
                    float2 imageMotionVector = calcImageMotionVector(path, params.frameDim, primaryHit);
                    deltaReflectionMotionVectors[pixel] = imageMotionVector;
                }

                path.terminate();
                return;
            }

            // For glass in reflections, force guide paths to always follow transmission/reflection based on albedos.
            // This is pretty hacky but works best our of the possible options.
            // Stable guide buffers are a necessity.
            if (bsdfProperties.isTransmissive && all(bsdfProperties.specularReflectionAlbedo <= bsdfProperties.specularTransmissionAlbedo))
            {
                // if total internal reflection happens just choose reflection
                if (isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
                    sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection | smoothMask & (uint)LobeType::SpecularReflection);
                else sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission | smoothMask & (uint)LobeType::SpecularTransmission);
            }
            else
            {
                sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection | smoothMask & (uint)LobeType::SpecularReflection);
            }
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeRayOrigin();

        // Hijack pdf that we don't need.
        float primaryHitDist = path.pdf;

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        path.pdf = primaryHitDist;

        if (!valid)
        {
            if (kOutputNRDAdditionalData)
            {
                writeNRDDeltaReflectionGuideBuffers(
                    outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength
                );
            }
            if (kOutputDeltaMotionVectors)
            {
                deltaReflectionMotionVectors[pixel] = 0.f;
            }
            path.terminate();
            return;
        }

        // Terminate if transmission lobe was chosen but volume absorption is too high
        // but store the previous vertex shading data.
        if (path.isTransmission())
        {
            // Fetch volume absorption from the material. This field only exist in basic materials for now.
            bool semiOpaque = false;
            if (gScene.materials.isBasicMaterial(sd.materialID))
            {
                BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
                // TODO: Expose this arbitrary value as a constant.
                semiOpaque = any(md.volumeAbsorption > 100.f);
            }

            if (semiOpaque)
            {
                const float3 emission = path.L;
                const float3 reflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);
                const float primaryHitDist = path.pdf;
                const float hitDist = float(path.sceneLength) - primaryHitDist;
                if (kOutputNRDAdditionalData)
                {
                    writeNRDDeltaReflectionGuideBuffers(
                        outputNRD,
                        kUseNRDDemodulation,
                        pixel,
                        reflectance,
                        emission,
                        bsdfProperties.guideNormal,
                        bsdfProperties.roughness,
                        float(path.sceneLength),
                        hitDist
                    );
                }
                if (kOutputDeltaMotionVectors)
                {
                    const HitInfo primaryHit = unpackHitInfo(vbuffer[pixel]);
                    // Place the current path origin at the terminal hit so calcImageMotionVector can project it.
                    path.origin = sd.posW;
                    float2 imageMotionVector = calcImageMotionVector(path, params.frameDim, primaryHit);
                    deltaReflectionMotionVectors[pixel] = imageMotionVector;
                }

                path.terminate();
                return;
            }
        }

        // Accumulate the image-space xform for this bounce.
        // This is used later to compute a virtual-image motion vector.
        // We assume flat reflectors/refractors 
        float3 originOffset = sd.posW - path.origin;
        float4x4 T;
        if (path.isTransmission())
        {
            float insideIoR = gScene.materials.evalIoR(sd.materialID);
            float rcpEta = sd.frontFacing ? (insideIoR / sd.IoR) : (sd.IoR / insideIoR);
            // For refractive (double sided) materials bsdfProperties.guideNormal always points in the same direction as the view vector.
            T = getRefractionXform(path.origin, viewDir, bsdfProperties.guideNormal, rcpEta, originOffset);
        }
        else
        {
            T = getReflectionXform(path.origin, viewDir, bsdfProperties.guideNormal, originOffset);
        }
        path.imageXform = mul(path.imageXform, T);
    }

    /** Handle hit on delta transmission materials.
        After handling the hit, a new scatter (delta transmission only) ray is generated or the path is terminated.
        Executed only for guide paths.
        \param[in,out] path The path state.
    */
    void handleDeltaTransmissionHit(inout PathState path)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Write out albedo/normal/posW on the first hit of non delta transmission BSDF lobe
        // - Sample scatter ray or terminate

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir);

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path))
        {
            float3 originOffset = sd.posW - path.origin;

            float4x4 originOffsetMat = float4x4(
                1.f, 0.f, 0.f, originOffset.x,
                0.f, 1.f, 0.f, originOffset.y,
                0.f, 0.f, 1.f, originOffset.z,
                0.f, 0.f, 0.f, 1.f
            );

            path.imageXform = mul(path.imageXform, originOffsetMat);

            return;
        }

        // Create texture sampler.
        let lod = createTextureSampler(path, isPrimaryHit);

        // Create material instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        let bsdfProperties = mi.getProperties(sd);

        const uint lobeTypes = mi.getLobeTypes(sd);

#if USE_RESTIRPT // allow computing specmvec for non-delta surface for ReSTIR PT tempora reuse
        bool isSmooth = bsdfProperties.roughness <= restirpt.specularMotionVectorRoughnessThreshold;
        // disable soft roughness for delta for ReSTIR specmvec because it conflicts with NRD denoiser output
        if (kOutputNRDData && path.isDeltaOnlyPath()) isSmooth = bsdfProperties.roughness == 0.f;
#else
        bool isSmooth = bsdfProperties.roughness == 0.f;
#endif

        uint smoothMask = uint(isSmooth) * uint(LobeType::All);

        // Terminate without the write-out if the path doesn't start as delta transmission.
        const bool hasDeltaTransmissionLobe = ((lobeTypes & (uint)LobeType::DeltaTransmission) != 0);
        if (isPrimaryHit && !hasDeltaTransmissionLobe)
        {
            if (kOutputNRDAdditionalData)
            {
                writeNRDDeltaTransmissionGuideBuffers(
                    outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, 0.f
                );
            }
            if (kOutputDeltaMotionVectors)
            {
                deltaTransmissionMotionVectors[pixel] = 0.f;
            }
            path.terminate();
            return;
        }

        if (isPrimaryHit)
        {
            // Add primary ray length to the path length.
            path.sceneLength += float16_t(length(sd.posW - path.origin));
        }
        else
        {
            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * bsdfProperties.emission;
        }

        // Terminate the delta transmission path.
        const bool lastVertex = hasFinishedSurfaceBounces(path);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        // Fetch volume absorption from the material. This field only exist in basic materials for now.
        bool semiOpaque = false;
        if (gScene.materials.isBasicMaterial(sd.materialID))
        {
            BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
            // TODO: Expose this arbitrary value as a constant.
            semiOpaque = any(md.volumeAbsorption > 100.f);
        }

        const MaterialType materialType = sd.mtl.getMaterialType();
        const bool hasDeltaLobes = (lobeTypes & (uint)LobeType::Delta) != 0;

        if (lastVertex || semiOpaque)
        {
            float3 emission = path.L;
            if (kOutputNRDAdditionalData)
            {
                writeNRDDeltaTransmissionGuideBuffers(
                    outputNRD,
                    kUseNRDDemodulation,
                    pixel,
                    getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties),
                    emission,
                    bsdfProperties.guideNormal,
                    bsdfProperties.roughness,
                    float(path.sceneLength),
                    sd.posW
                );
            }
            if (kOutputDeltaMotionVectors)
            {
                const HitInfo primaryHit = unpackHitInfo(vbuffer[pixel]);
                // Place the current path origin at the terminal hit so calcImageMotionVector can project it.
                path.origin = sd.posW;
                float2 imageMotionVector = calcImageMotionVector(path, params.frameDim, primaryHit);
                deltaTransmissionMotionVectors[pixel] = imageMotionVector;
            }

            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeRayOrigin();

        // Set the active lobes only to delta transmission.
        sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission | smoothMask & (uint)LobeType::SpecularTransmission);

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        // Delta transmission was not possible, fallback to delta reflection if it's allowed.
        if (!valid && isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
        {
            sd.mtl.setActiveLobes( (uint)LobeType::DeltaTransmission | (uint)LobeType::DeltaReflection | smoothMask & (uint)LobeType::SpecularTransmission | smoothMask & (uint)LobeType::SpecularReflection );
            valid = generateScatterRay(sd, mi, path);
        }

        if (!valid)
        {
            float3 emission = path.L;
            if (kOutputNRDAdditionalData)
            {
                writeNRDDeltaTransmissionGuideBuffers(
                    outputNRD,
                    kUseNRDDemodulation,
                    pixel,
                    getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties),
                    emission,
                    bsdfProperties.guideNormal,
                    bsdfProperties.roughness,
                    float(path.sceneLength),
                    sd.posW
                );
            }
            if (kOutputDeltaMotionVectors)
            {
                const HitInfo primaryHit = unpackHitInfo(vbuffer[pixel]);
                // Place the current path origin at the terminal hit so calcImageMotionVector can project it.
                path.origin = sd.posW;
                float2 imageMotionVector = calcImageMotionVector(path, params.frameDim, primaryHit);
                deltaTransmissionMotionVectors[pixel] = imageMotionVector;
            }

            path.terminate();
            return;
        }

        // Accumulate the image-space xform for this bounce.
        // This is used later to compute a virtual-image motion vector for the denoiser.
        // We assume flat reflectors/refractors
        float3 originOffset = sd.posW - path.origin;
        float4x4 T;
        if (path.isTransmission())
        {
            float insideIoR = gScene.materials.evalIoR(sd.materialID);
            float rcpEta = sd.frontFacing ? (insideIoR / sd.IoR) : (sd.IoR / insideIoR);
            // For refractive (double sided) materials bsdfProperties.guideNormal always points in the same direction as the view vector.
            T = getRefractionXform(path.origin, viewDir, bsdfProperties.guideNormal, rcpEta, originOffset);
        }
        else
        {
            T = getReflectionXform(path.origin, viewDir, bsdfProperties.guideNormal, originOffset);
        }
        path.imageXform = mul(path.imageXform, T);
    }
}
