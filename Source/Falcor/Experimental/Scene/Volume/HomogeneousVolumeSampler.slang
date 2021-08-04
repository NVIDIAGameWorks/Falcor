/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
import Scene.Scene;
__exported import Utils.Sampling.SampleGeneratorInterface;
import Phasefunctions;

/** Helper class for sampling homogeneous volumes.
*/
struct HomogeneousVolumeSampler
{
    float3 sigmaA;          ///< Absorption coefficient.
    float3 sigmaS;          ///< Scattering coefficient.
    float3 sigmaT;          ///< Extinction coefficient (sigmaA + sigmaS).
    float g;                ///< Scattering anisotropy.
    float3 channelProbs;    ///< Scattering channel probabilities.
    uint channel;           ///< Sampled scattering channel.

    /** Create a new sampler from medium properties of a material in the scene.
        \return Returns the sampler.
    */
    static HomogeneousVolumeSampler create(uint materialID)
    {
        MaterialData md = gScene.getMaterial(materialID);
        return create(md.volumeAbsorption, md.volumeScattering, md.volumeAnisotropy);
    }

    /** Create a new sampler from medium properties.
        \param[in] sigmaA Absorption coefficient.
        \param[in] sigmaS Scattering coefficient.
        \param[in] g Scattering anisotropy.
        \return Returns the sampler.
    */
    static HomogeneousVolumeSampler create(float3 sigmaA, float3 sigmaS, float g)
    {
        HomogeneousVolumeSampler vs = {};
        vs.sigmaA = sigmaA;
        vs.sigmaS = sigmaS;
        vs.sigmaT = sigmaA + sigmaS;
        vs.g = g;
        return vs;
    }

    /** Compute channel sampling probabilities given a path throughput.
        \param[in] Path throughput.
    */
    [mutating] void computeChannelProbs(float3 thp)
    {
        // Compute albedo (set to 1 if no scattering for hero selection).
        const float3 albedo = float3(
            sigmaS.r > 0.f ? sigmaS.r / sigmaT.r : 1.f,
            sigmaS.g > 0.f ? sigmaS.g / sigmaT.g : 1.f,
            sigmaS.b > 0.f ? sigmaS.b / sigmaT.b : 1.f,
        );

        // Compute probabilities for selecting RGB channels for scatter distance sampling.
        channelProbs = thp * albedo;
        const float channelProbsSum = channelProbs.r + channelProbs.g + channelProbs.b;
        if (channelProbsSum < 1e-8f)
        {
            channelProbs = float3(1.f) / 3.f;
        }
        else
        {
            channelProbs /= channelProbsSum;
        }
    }

    /** Sample a scattering distance.
        \param[in] thp Current path throughput (used to compute channel sampling probabilities).
        \param[in,out] sg Sample generator.
        \return Returns the sampled distance.
    */
    [mutating] float sampleDistance<S : ISampleGenerator>(float3 thp, inout S sg)
    {
        computeChannelProbs(thp);

        // Sample RGB channel.
        const float xi = sampleNext1D(sg);
        if (xi < channelProbs.r)
        {
            channel = 0;
        }
        else if (xi < channelProbs.r + channelProbs.g)
        {
            channel = 1;
        }
        else
        {
            channel = 2;
        }

        const float u = sampleNext1D(sg);
        return -log(1.f - u) / sigmaS[channel];
    }

    /** Evaluate the transmittance based on wheter a scattering event has happened or not.
        Note: This method depends on `sampleDistance()` being called first.
        \param[in] distance Distance to evaluate transmittance for.
        \param[in] hasScattered True if a scattering has happened.
        \return Returns the transmittance value.
    */
    float3 evalTransmittance(float distance, bool hasScattered)
    {
        const float3 Tr = exp(-sigmaT * distance);
        const float3 Trs = exp(-sigmaS * distance);
        const float3 density = hasScattered ? Trs * sigmaS : Trs;
        const float pdf = dot(channelProbs, density);
        return hasScattered ? Tr * sigmaS / pdf : Tr / pdf;
    }

    /** Sample a scattering direction.
        \param[in] rayDir Ray direction.
        \param[input] sg Sample generator.
        \param[out] scatterDir Sampled scatter direction.
        \return Returns true if successful.
    */
    bool samplePhaseFunction<S : ISampleGenerator>(const float3 rayDir, inout S sg, out float3 scatterDir)
    {
        scatterDir = samplePhaseHenyeyGreenstein(rayDir, g, sampleNext2D(sg));
        return true;
    }

    /** Evaluate the directional probability for sampling a scattering direction.
        \param[in] rayDir Ray direction.
        \param[in] scatterDir Scatter direction.
        \return Returns the directional probability.
    */
    float evalPhaseFunction(const float3 rayDir, const float3 scatterDir)
    {
        return evalPhaseHenyeyGreenstein(dot(-rayDir, scatterDir), g);
    }
}
