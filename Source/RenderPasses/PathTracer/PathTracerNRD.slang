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
import RenderPasses.Shared.Denoising.NRDBuffers;
import RenderPasses.Shared.Denoising.NRDConstants;
import RenderPasses.Shared.Denoising.NRDHelpers;
import LoadShadingData;
import PathTracer;
import NRDHelpers;

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
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        // Create BSDF instance and query its properties.
        let bsdf = gScene.materials.getBSDF(sd, lod);
        let bsdfProperties = bsdf.getProperties(sd);

        // Query BSDF lobes.
        const uint lobes = bsdf.getLobes(sd);
        const bool hasDeltaLobes = (lobes & (uint)LobeType::Delta) != 0;

        const MaterialType materialType = sd.mtl.getMaterialType();

        if (isPrimaryHit)
        {
            // Terminate without the write-out if the path doesn't start as delta reflection.
            bool hasDeltaReflectionLobe = ((lobes & (uint)LobeType::DeltaReflection) != 0);
            if (!hasDeltaReflectionLobe)
            {
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
                path.terminate();
                return;
            }

            // Add primary ray length to the path length.
            float primaryHitDist = length(sd.posW - path.origin);
            path.sceneLength += float16_t(primaryHitDist);
            // Hijack pdf that we don't need.
            path.pdf += primaryHitDist;

            // Set the active lobes only to delta reflection on the first bounce.
            sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection);
        }
        else
        {
            // Use path's radiance field to accumulate emission along the path since the radiance is not used for denoiser guide paths.
            // No need for accumulating emission at the primary hit since the primary hit emission is coming from GBuffer.
            path.L += path.thp * bsdfProperties.emission;

            // Terminate after scatter ray on last vertex has been processed or non-delta lobe exists.
            const bool lastVertex = hasFinishedSurfaceBounces(path);
            const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;
            const bool isEmissive = any(bsdfProperties.emission > 0.f);

            if (lastVertex || hasNonDeltaLobes || isEmissive)
            {
                const float3 emission = path.L;
                const float3 reflectance = getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties);
                const float primaryHitDist = path.pdf;
                const float hitDist = float(path.sceneLength) - primaryHitDist;
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, reflectance, emission, sd.N, bsdfProperties.roughness, float(path.sceneLength), hitDist);

                path.terminate();
                return;
            }

            // For glass in reflections, force guide paths to always follow transmission/reflection based on albedos.
            // This is pretty hacky but works best our of the possible options.
            // Stable guide buffers are a necessity.
            if (bsdfProperties.isTransmissive && all(bsdfProperties.specularReflectionAlbedo <= bsdfProperties.specularTransmissionAlbedo))
            {
                sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission);
            }
            else
            {
                sd.mtl.setActiveLobes((uint)LobeType::DeltaReflection);
            }
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeNewRayOrigin();

        // Hijack pdf that we don't need.
        float primaryHitDist = path.pdf;

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, bsdf, path);

        path.pdf = primaryHitDist;

        if (!valid)
        {
            writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, kNRDInvalidPathLength);
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
                writeNRDDeltaReflectionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, reflectance, emission, sd.N, bsdfProperties.roughness, float(path.sceneLength), hitDist);

                path.terminate();
                return;
            }
        }
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
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const uint2 pixel = path.getPixel();
        const float3 viewDir = -path.dir;

        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir, isPrimaryHit, lod);

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return;

        // Create BSDF instance and query its properties.
        let bsdf = gScene.materials.getBSDF(sd, lod);
        let bsdfProperties = bsdf.getProperties(sd);

        const uint lobes = bsdf.getLobes(sd);

        // Terminate without the write-out if the path doesn't start as delta transmission.
        const bool hasDeltaTransmissionLobe = ((lobes & (uint)LobeType::DeltaTransmission) != 0);
        if (isPrimaryHit && !hasDeltaTransmissionLobe)
        {
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, 0.f, 0.f, viewDir, 0.f, kNRDInvalidPathLength, 0.f);
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
        const bool hasNonDeltaLobes = (lobes & (uint)LobeType::NonDelta) != 0;

        // Fetch volume absorption from the material. This field only exist in basic materials for now.
        bool semiOpaque = false;
        if (gScene.materials.isBasicMaterial(sd.materialID))
        {
            BasicMaterialData md = gScene.materials.getBasicMaterialData(sd.materialID);
            // TODO: Expose this arbitrary value as a constant.
            semiOpaque = any(md.volumeAbsorption > 100.f);
        }

        const MaterialType materialType = sd.mtl.getMaterialType();
        const bool hasDeltaLobes = (lobes & (uint)LobeType::Delta) != 0;

        if (lastVertex || semiOpaque)
        {
            float3 emission = path.L;
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties), emission, sd.N, bsdfProperties.roughness, float(path.sceneLength), sd.posW);

            path.terminate();
            return;
        }

        // Compute origin for rays traced from this path vertex.
        path.origin = sd.computeNewRayOrigin();

        // Set the active lobes only to delta transmission.
        sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission);

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, bsdf, path);

        // Delta transmission was not possible, fallback to delta reflection if it's allowed.
        if (!valid && isDeltaReflectionAllowedAlongDeltaTransmissionPath(sd))
        {
            sd.mtl.setActiveLobes((uint)LobeType::DeltaTransmission | (uint)LobeType::DeltaReflection);
            valid = generateScatterRay(sd, bsdf, path);
        }

        if (!valid)
        {
            float3 emission = path.L;
            writeNRDDeltaTransmissionGuideBuffers(outputNRD, kUseNRDDemodulation, pixel, getMaterialReflectanceForDeltaPaths(materialType, hasDeltaLobes, sd, bsdfProperties), emission, sd.N, bsdfProperties.roughness, float(path.sceneLength), sd.posW);

            path.terminate();
            return;
        }
    }
}
