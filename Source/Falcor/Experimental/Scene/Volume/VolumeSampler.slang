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
import Utils.Helpers;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGeneratorInterface;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.HashUtils;
import RenderPasses.Shared.PathTracer.PixelStats;
import PhaseFunctions;
import VolumeSamplerParams;

#ifndef VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR
// VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR must be defined in order to use this module.
#error "VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR not defined!"
#endif

/** Helper class for sampling volumes in the scene.
    Note: For simplicity, this sampler only uses the first volume in the scene.
*/
struct VolumeSampler
{
    static const TransmittanceEstimator kTransmittanceEstimator = TransmittanceEstimator(VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR);

    uint4 _dummy;

    struct DistanceSample
    {
        float t;    ///< Sampled distance.
        float3 thp; ///< Througput.
    };

    /** Checks if a ray potentially intersects a volume.
        \param[in] rayOrigin Ray origin.
        \param[in] rayDir Ray direction.
        \param[in] minT Ray minimum t.
        \param[in] maxT Ray maximum t.
        \return Returns true if a volume potentially intersects the ray.
    */
    bool intersectsVolumes(const float3 rayOrigin, const float3 rayDir, const float minT, const float maxT)
    {
        if (gScene.getVolumeCount() == 0) return false;
        float2 nearFar;
        return intersectVolume(gScene.getVolume(0), rayOrigin, rayDir, minT, maxT, nearFar);
    }

    /** Evaluate transmittance along a ray through volumes.
        \param[in] rayOrigin Ray origin.
        \param[in] rayDir Ray direction.
        \param[in] minT Ray minimum t.
        \param[in] maxT Ray maximum t.
        \param[in,out] sg Sample generator.
        \return Returns the transmittance.
    */
    float evalTransmittance<S : ISampleGenerator>(const float3 rayOrigin, const float3 rayDir, const float minT, const float maxT, inout S sg)
    {
        if (gScene.getVolumeCount() == 0) return 1.f;

        Volume volume = gScene.getVolume(0);
        if (!volume.hasDensityGrid()) return 1.f;

        float2 nearFar;
        if (!intersectVolume(volume, rayOrigin, rayDir, minT, maxT, nearFar)) return 1.f;

        TinyUniformSampleGenerator tsg = TinyUniformSampleGenerator::create(jenkinsHash(sg.next()));

        switch (kTransmittanceEstimator)
        {
        case TransmittanceEstimator::DeltaTracking:
            return evalTransmittanceDeltaTracking(volume, rayOrigin, rayDir, nearFar, tsg);
        case TransmittanceEstimator::RatioTracking:
            return evalTransmittanceRatioTracking(volume, rayOrigin, rayDir, nearFar, tsg);
        default:
            return 1.f;
        }
    }

    /** Sample a scattering distance along a ray through volumes.
        \param[in] rayOrigin Ray origin.
        \param[in] rayDir Ray direction.
        \param[in] minT Ray minimum t.
        \param[in] maxT Ray maximum t.
        \param[in,out] sg Sample generator.
        \param[out] ds Distance sample.
        \return Returns true if a valid scattering distance was sampled.
    */
    bool sampleDistance<S : ISampleGenerator>(const float3 rayOrigin, const float3 rayDir, const float minT, const float maxT, inout S sg, out DistanceSample ds)
    {
        ds = {};

        if (gScene.getVolumeCount() == 0) return false;

        Volume volume = gScene.getVolume(0);
        if (!volume.hasDensityGrid()) return false;

        float2 nearFar;
        if (!intersectVolume(volume, rayOrigin, rayDir, minT, maxT, nearFar)) return false;

        TinyUniformSampleGenerator tsg = TinyUniformSampleGenerator::create(jenkinsHash(sg.next()));
        return sampleDistanceDeltaTracking(volume, rayOrigin, rayDir, nearFar, tsg, ds);
    }

    /** Sample a scattering direction.
        \param[in] rayDir Ray direction.
        \param[input] sg Sample generator.
        \param[out] scatterDir Sampled scatter direction.
        \return Returns true if successful.
    */
    bool samplePhaseFunction<S : ISampleGenerator>(const float3 rayDir, inout S sg, out float3 scatterDir)
    {
        scatterDir = {};

        if (gScene.getVolumeCount() == 0) return false;

        Volume volume = gScene.getVolume(0);
        scatterDir = samplePhaseHenyeyGreenstein(rayDir, volume.data.anisotropy, sampleNext2D(sg));

        return true;
    }

    /** Evaluate the directional probability for sampling a scattering direction.
        \param[in] rayDir Ray direction.
        \param[in] scatterDir Scatter direction.
        \return Returns the directional probability.
    */
    float evalPhaseFunction(const float3 rayDir, const float3 scatterDir)
    {
        if (gScene.getVolumeCount() == 0) return 0.f;

        Volume volume = gScene.getVolume(0);
        return evalPhaseHenyeyGreenstein(dot(-rayDir, scatterDir), volume.data.anisotropy);
    }

    float lookupDensity(const Volume volume, const float3 pos, const float3 u, const Grid densityGrid, inout Grid::Accessor densityAccessor)
    {
        logVolumeLookup();
        return volume.data.densityScale * densityGrid.lookupStochasticWorld(mul(float4(pos, 1.f), volume.data.invTransform).xyz, u, densityAccessor);
    }

    bool intersectVolume(const Volume volume, const float3 rayOrigin, const float3 rayDir, const float minT, const float maxT, out float2 nearFar)
    {
        // Intersect with volume bounds and get intersection interval along the view ray.
        AABB bounds = volume.getBounds();
        intersectRayAABB(rayOrigin, rayDir, bounds.minPoint, bounds.maxPoint, nearFar);
        nearFar.x = max(nearFar.x, minT);
        nearFar.y = min(nearFar.y, maxT);
        return nearFar.x < nearFar.y;
    }

    float evalTransmittanceDeltaTracking<S : ISampleGenerator>(const Volume volume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(volume.getDensityGrid(), densityGrid);
        Grid::Accessor densityAccessor = densityGrid.createAccessor();

        float Tr = 1.f;
        const float invMajorant = 1.f / (volume.data.densityScale * densityGrid.getMaxValue());

        // Delta tracking.
        float t = nearFar.x;
        while (t < nearFar.y)
        {
            t -= log(1 - sampleNext1D(sg)) * invMajorant;
            const float d = lookupDensity(volume, rayOrigin + t * rayDir, sampleNext3D(sg), densityGrid, densityAccessor);
            Tr *= 1.f - max(0.f, d * invMajorant);
            // Russian roulette.
            if (sampleNext1D(sg) < d * invMajorant) return 0.f;
        }
        return 1.f;
    }

    float evalTransmittanceRatioTracking<S : ISampleGenerator>(const Volume volume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg)
    {
        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(volume.getDensityGrid(), densityGrid);
        Grid::Accessor densityAccessor = densityGrid.createAccessor();

        float Tr = 1.f;
        const float invMajorant = 1.f / (volume.data.densityScale * densityGrid.getMaxValue());

        // Ratio tracking.
        float t = nearFar.x;
        while (t < nearFar.y)
        {
            t -= log(1 - sampleNext1D(sg)) * invMajorant;
            const float d = lookupDensity(volume, rayOrigin + t * rayDir, sampleNext3D(sg), densityGrid, densityAccessor);
            Tr *= 1.f - max(0.f, d * invMajorant);
            if (Tr < 0.1f)
            {
                // Russian roulette.
                const float prob = 1 - Tr;
                if (sampleNext1D(sg) < prob) return 0.f;
                Tr /= 1.f - prob;
            }
        }

        return Tr;
    }

    bool sampleDistanceDeltaTracking<S : ISampleGenerator>(const Volume volume, const float3 rayOrigin, const float3 rayDir, const float2 nearFar, inout S sg, out DistanceSample ds)
    {
        ds = {};

        // Setup access to density grid.
        Grid densityGrid;
        gScene.getGrid(volume.getDensityGrid(), densityGrid);
        Grid::Accessor densityAccessor = densityGrid.createAccessor();

        const float invMajorant = 1.f / (volume.data.densityScale * densityGrid.getMaxValue());

        // Delta tracking.
        float t = nearFar.x;
        while (t < nearFar.y)
        {
            t -= log(1 - sampleNext1D(sg)) * invMajorant;
            const float d = lookupDensity(volume, rayOrigin + t * rayDir, sampleNext3D(sg), densityGrid, densityAccessor);
            // Scatter on real collision.
            if (sampleNext1D(sg) < d * invMajorant)
            {
                ds.t = t;
                ds.thp = volume.data.albedo;
                return true;
            }
        }

        return false;
    }
};
