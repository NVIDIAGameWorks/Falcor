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

/** Interface for evaluating irradiance using an RTXGI probe volume.

    Import the module in your shader and bind its parameter block.
    Evaluate irradiance using `gRTXGIVolume.evalIrradiance()`.
    See also the other functions for querying the probe volume.
*/

#include "RTXGIDefines.slangh"
#include "rtxgi/shaders/ddgi/Irradiance.hlsl"
#include "Utils/Math/MathConstants.slangh"

struct RTXGIVolume
{
    static const uint kProbeStateActive = RTXGI_DDGI_PROBE_STATE_ACTIVE;
    static const uint kProbeStateInactive = RTXGI_DDGI_PROBE_STATE_INACTIVE;
    static const uint kNumFixedRays = RTXGI_DDGI_NUM_FIXED_RAYS;

    DDGIVolumeDescGPUPacked volumePacked;
    DDGIVolumeResources resources;

    /** Computhe the surface bias to use for irradiance lookups.
        \param[in] normal Normalized surface normal.
        \param[in] view Normalized view direction.
        \return Surface bias.
    */
    float3 getSurfaceBias(float3 normal, float3 view)
    {
        const DDGIVolumeDescGPU volume = getVolumeDescGPU();
        return normal * volume.probeNormalBias + view * volume.probeViewBias;
    }

    /** Evaluates the irradiance (flux per unit area) arriving at a surface in the world.
        The irradiance is computed by a filtered lookup in the probe volume.
        \param[in] posW World-space position.
        \param[in] normal Normalized surface normal.
        \param[in] view Normalized view direction.
        \param[in] useBlending Use volume blend weight.
        \return Irradiance (linear).
    */
    float3 evalIrradiance(float3 posW, float3 normal, float3 view, bool useBlending = false)
    {
        const DDGIVolumeDescGPU volume = getVolumeDescGPU();

        float3 irradiance = float3(0);

        // Compute volume blending weight
        float volumeBlendWeight = 1.f;
        if (useBlending) volumeBlendWeight = DDGIGetVolumeBlendWeight(posW, volume);

        if (volumeBlendWeight > 0.f)
        {
            // Get irradiance from the probe volume.
            float3 surfaceBias = getSurfaceBias(normal, view);
            irradiance = DDGIGetVolumeIrradiance(posW, surfaceBias, normal, volume, resources);

            // Attenuate irradiance by the blend weight.
            irradiance *= volumeBlendWeight;
        }

        return irradiance;
    }

    /** Evaluates the radiance outgoing at a surface in the world.
        We assume a perfect Lambertian diffuse lobe, as that is the only
        case that can be correctly factored out and multiplied by preintegrated irradiance.
        Note that this is an approximation to actual directional diffuse BSDF lobe used elsewhere.
        \param[in] posW World-space position.
        \param[in] normal Normalized surface normal.
        \param[in] view Normalized view direction.
        \param[in] albedo Diffuse albedo of the surface.
        \param[in] useBlending Use volume blend weight.
        \return Radiance (linear).
    */
    float3 evalRadiance(float3 posW, float3 normal, float3 view, float3 albedo, bool useBlending = false)
    {
        float3 E = evalIrradiance(posW, normal, view, useBlending);
        float3 L = (float)M_1_PI * albedo * E;
        return L;
    }

    /** Computes the 3D grid coordinates for the probe at the given probe index.
    */
    int3 getProbeCoords(int probeIndex)
    {
        const DDGIVolumeDescGPU volume = getVolumeDescGPU();

        return DDGIGetProbeCoords(probeIndex, volume);
    }

    /** Returns the probe state classifier for a given probe index.
    */
    uint getProbeState(int probeIndex)
    {
        const DDGIVolumeDescGPU volume = getVolumeDescGPU();

        uint state = DDGILoadProbeState(probeIndex, resources.probeData, volume);
        return state;
    }

    /** Computes the world space position of the probe at the given probe index.
    */
    float3 getProbeWorldPosition(int probeIndex)
    {
        const DDGIVolumeDescGPU volume = getVolumeDescGPU();

        int3 probeCoords = DDGIGetProbeCoords(probeIndex, volume);
        return DDGIGetProbeWorldPosition(probeCoords, volume, resources.probeData);
    }

    /** Evaluates the irradiance for a given normal direction and probe index.
        The irradiance is directly sampled from the irradiance texture at the given probe.
        \param[in] probeIndex Probe index.
        \param[in] normal Normalized surface normal.
    */
    float3 evalProbeIrradiance(int probeIndex, float3 normal)
    {
        const DDGIVolumeDescGPU volume = getVolumeDescGPU();

        // Convert sampling direction to probe texture coords.
        float2 coords = DDGIGetOctahedralCoordinates(normal); // Normalized octahedral coordinates in [-1,1].
        float2 uv = DDGIGetProbeUV(probeIndex, coords, volume.probeNumIrradianceTexels, volume); // Tex coords in irradiance texture

        // Sample the probe irradiance texture.
        float3 irradiance = resources.probeIrradiance.SampleLevel(resources.bilinearSampler, uv, 0).rgb;

        // Decode the tone curve.
        float3 exponent = volume.probeIrradianceEncodingGamma;
        irradiance = pow(irradiance, exponent);
        irradiance *= (0.5f * RTXGI_PI); // Factored out of the probes

        return irradiance;
    }


    /** Unpackes the member volume descriptor to be passed into the various DDGI calls.
        \return The unpacked VolumeDescGPU
    */
    DDGIVolumeDescGPU getVolumeDescGPU()
    {
        return UnpackDDGIVolumeDescGPU(volumePacked); ;
    }
};

ParameterBlock<RTXGIVolume> gRTXGIVolume;
