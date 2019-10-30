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
#include "Utils/Math/MathConstants.slang"

import Utils.Sampling.SampleGenerator;
import Experimental.Scene.Lights.LightCollection;
import Experimental.Scene.Lights.EmissiveLightSamplerHelpers;
import Experimental.Scene.Lights.EmissiveLightSamplerInterface;

/** Emissive light sampler using uniform sampling of the emissive triangles.

    The sampler implements the IEmissiveLightSampler interface (see
    EmissiveLightSamplerInterface.slang for usage information).

    The struct wraps a LightCollection that stores the pre-processed lights.
    The program should instantiate the struct below. See EmissiveLightSampler.slang.
*/
struct EmissiveUniformSampler : IEmissiveLightSampler
{
    LightCollection     _lights;        ///< The light sources.

    /** Draw a single light sample.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in,out] sg Sample generator.
        \param[out] ls Light sample. If ls.pdf == 0.0 the sample is invalid and should not be used.
    */
    void sampleLight(const float3 posW, const float3 normalW, inout SampleGenerator sg, out TriangleLightSample ls)
    {
        ls = {};
    #if _NUM_MESH_LIGHTS == 0
        return;
    #else
        // Randomly pick a triangle out of the global list with uniform probability.
        float uLight = sampleNext1D(sg);
        uint triangleCount = _lights.triangleCount;
        uint triangleIndex = min((uint)(uLight * triangleCount), triangleCount - 1); // Safety precaution as the result of the multiplication may be rounded to triangleCount even if uLight < 1.0 when triangleCount is large.
        float triangleSelectionPdf = 1.0f / ((float)triangleCount);

        // Sample the triangle uniformly.
        float2 u = sampleNext2D(sg);
        sampleTriangle(_lights, posW, triangleIndex, u, ls);

        // The final probability density is the product of the sampling probabilities.
        ls.pdf *= triangleSelectionPdf;

    #endif // _NUM_MESH_LIGHTS
    }

    /** Evaluate the PDF at a shading point given a hit point on an emissive triangle.
        \param[in] posW Shading point in world space.
        \param[in] normalW Normal at the shading point in world space.
        \param[in] hit Triangle hit data.
        \return Probability density with respect to solid angle at the shading point.
    */
    float evalPdf(float3 posW, float3 normalW, const TriangleHit hit)
    {
    #if _NUM_MESH_LIGHTS == 0
        return 0.0f;
    #else
        // Lights are chosen uniformly so the selection probability is just one over the number of lights.
        float triangleSelectionPdf = 1.0f / ((float)_lights.triangleCount);

        // Compute triangle sampling probability with respect to solid angle from the shading point.
        float trianglePdf = evalTrianglePdf(posW, hit);

        // The final probability density is the product of the sampling probabilities.
        return triangleSelectionPdf * trianglePdf;

    #endif // !_NUM_MESH_LIGHTS
    }
};
