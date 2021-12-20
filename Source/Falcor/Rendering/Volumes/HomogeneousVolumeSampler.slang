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
__exported import Scene.Material.HomogeneousVolumeData;
__exported import Utils.Sampling.SampleGeneratorInterface;

/** Helper class for sampling homogeneous volumes.
*/
struct HomogeneousVolumeSampler
{
    struct DistanceSample
    {
        float t;                ///< Scatter distance.
        float3 sigmaS;          ///< Scattering coefficient.
        float3 sigmaT;          ///< Extinction coefficient.
        float3 channelProbs;    ///< Channel selection probabilities.

        /** Evaluate the path throughput based on whether a scattering event has happened or not.
            \param[in] hasScattered True if a scattering has happened.
            \param[in] surfaceT Distance to the surface if hasScattered == false.
            \return Returns the throughput.
        */
        float3 evalThroughput(const bool hasScattered, const float surfaceT)
        {
            const float evalT = hasScattered ? t : surfaceT;
            const float3 Tr = exp(-sigmaT * evalT);
            const float3 Trs = exp(-sigmaS * evalT);
            const float3 density = hasScattered ? Trs * sigmaS : Trs;
            const float pdf = dot(channelProbs, density);
            return hasScattered ? Tr * sigmaS / pdf : Tr / pdf;
        }
    };

    /** Sample a scattering distance.
        \param[in] sigmaA Absorption coefficient.
        \param[in] sigmaS Scattering coefficient.
        \param[in] thp Current path throughput (used to compute channel sampling probabilities).
        \param[in,out] sg Sample generator.
        \param[out] ds Distance sample.
        \return Returns true if a scattering distance was sampled (false if medium does not scatter).
    */
    static bool sampleDistance<S : ISampleGenerator>(const float3 sigmaA, const float3 sigmaS, const float3 thp, inout S sg, out DistanceSample ds)
    {
        if (all(sigmaS == 0.f)) return false;

        const float3 sigmaT = sigmaA + sigmaS;

        // Compute albedo (set to 1 if no scattering for hero selection).
        const float3 albedo = float3(
            sigmaS.r > 0.f ? sigmaS.r / sigmaT.r : 1.f,
            sigmaS.g > 0.f ? sigmaS.g / sigmaT.g : 1.f,
            sigmaS.b > 0.f ? sigmaS.b / sigmaT.b : 1.f,
        );

        // Compute probabilities for selecting RGB channels for scatter distance sampling.
        float3 channelProbs = thp * albedo;
        const float channelProbsSum = channelProbs.r + channelProbs.g + channelProbs.b;
        if (channelProbsSum < 1e-8f)
        {
            channelProbs = float3(1.f) / 3.f;
        }
        else
        {
            channelProbs /= channelProbsSum;
        }

        // Sample RGB channel.
        const float xi = sampleNext1D(sg);
        uint channel;
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
        const float t = -log(1.f - u) / sigmaS[channel];

        // Return distance sample.
        ds.t = t;
        ds.sigmaS = sigmaS;
        ds.sigmaT = sigmaT;
        ds.channelProbs = channelProbs;

        return true;
    }

    /** Sample a scattering distance.
        \param[in] vd Medium properties.
        \param[in] thp Current path throughput (used to compute channel sampling probabilities).
        \param[in,out] sg Sample generator.
        \param[out] ds Distance sample.
        \return Returns true if a scattering distance was sampled (false if medium does not scatter).
    */
    static bool sampleDistance<S : ISampleGenerator>(const HomogeneousVolumeData vd, const float3 thp, inout S sg, out DistanceSample ds)
    {
        return sampleDistance(vd.sigmaA, vd.sigmaS, thp, sg, ds);
    }

    /** Evaluate transmittance through a homogeneous medium.
        \param[in] sigmaT Extinction coefficient.
        \param[in] distance Distance through the medium.
        \return Returns the transmittance.
    */
    static float3 evalTransmittance(const float3 sigmaT, const float distance)
    {
        return exp(-distance * sigmaT);
    }

    /** Evaluate transmittance through a homogeneous medium.
        \param[in] vd Medium properties.
        \param[in] distance Distance through the medium.
        \return Returns the transmittance.
    */
    static float3 evalTransmittance(const HomogeneousVolumeData vd, const float distance)
    {
        return evalTransmittance(vd.sigmaA + vd.sigmaS, distance);
    }
}
