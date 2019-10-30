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

 /** Shared utility functions for emissive light sampler implementations.

     These functions rely on LightCollection, which is a container holding the
     global list of all emissive triangles in the scene.

     TODO: It would've been nice if we could have these functions in the interface
     struct (representing the "base" class that "derived" classes can use).
 */
#include "HostDeviceSharedMacros.h"
#include "Utils/Math/MathConstants.slang"

import Shading;
import Utils.Math.MathHelpers;
import Experimental.Scene.Lights.LightCollection;
import Experimental.Scene.Lights.EmissiveLightSamplerInterface;

/** Samples a single triangle uniformly and evaluates the probability density function.

    \param[in] lights The light collection.
    \param[in] posW Shading point in world space.
    \param[in] triangleIndex Triangle index of sampled triangle.
    \param[in] u Uniform random number (2D).
    \param[out] ls Light sample (ls.pdf = 0 indicates an invalid sample).
*/
void sampleTriangle(const LightCollection lights, const float3 posW, uint triangleIndex, float2 u, out TriangleLightSample ls)
{
    ls = {};
#if _NUM_MESH_LIGHTS == 0
    return;
#else
    // Sample the triangle uniformly.
    const float3 barycentrics = sample_triangle(u);
    ls.posW = lights.getPosition(triangleIndex, barycentrics);

    // Compute light vector and squared distance.
    float3 toLight = ls.posW - posW; // Unnormalized light vector
    const float distSqr = max(FLT_MIN, dot(toLight, toLight)); // Clamp to avoid NaNs below
    ls.distance = sqrt(distSqr);
    ls.dir = toLight / ls.distance; // Note: toLight can be zero.

    // Get triangle normal and associated quantities.
    ls.normalW = lights.getTriangleData(triangleIndex).normal;
    float triangleArea = lights.getTriangleData(triangleIndex).area;

    // Reject sample if back-facing.
    float cosTheta = dot(ls.normalW, -ls.dir);
    if (cosTheta <= 0.f) return; // ls.pdf==0 (and ls.Le==0) makes the sample invalid.

    // Evaluate emitted radiance.
    ls.Le = lights.getEmissive(triangleIndex, barycentrics);

    // Compute probability density with respect to solid angle from the shading point.
    // The farther away the light is and the larger the angle it is at, the larger the pdf becomes. The probability goes to infinity in the limit.
    // Note: Guard against div-by-zero here by clamping.
    float denom = max(FLT_MIN, cosTheta * triangleArea);
    ls.pdf = distSqr / denom;

    // TODO: We can simplify the expressions by using the unnormalized quantities for computing the pdf.
    // TODO: Look at SASS to understand the tradeoffs and instruction count for different varieties. What return values do we really need?
    //ls.pdf = -2.f * distSqr * ls.distance / (dot(N, L);    // Optimized (except N would have to be properly flipped above, before normalW is computed).
#endif // !_NUM_MESH_LIGHTS
}

/** Evaluates the PDF for a light sample given a hit point on an emissive triangle.
    \param[in] posW Shading point in world space.
    \param[in] hit Triangle hit data.
    \return Probability density with respect to solid angle at the shading point.
*/
float evalTrianglePdf(const float3 posW, const TriangleHit hit)
{
#if _NUM_MESH_LIGHTS == 0
    return 0.f;
#else
    // Compute light vector and squared distance.
    float3 toLight = hit.posW - posW; // Unnormalized light vector
    const float distSqr = dot(toLight, toLight);
    if (distSqr <= FLT_MIN) return 0.f; // Avoid NaNs below
    float3 L = toLight / sqrt(distSqr);

    // Cosine of angle between the light's normal and the light vector (flip L since it points towards the light).
    float cosTheta = dot(hit.normalW, -L);
    if (cosTheta <= 0.f) return 0.f;

    // Compute probability density with respect to solid angle from the shading point.
    // The farther away the light is and the larger the angle it is at, the larger the pdf becomes. The probability goes to infinity in the limit.
    // Note: Guard against div-by-zero here by clamping.
    // TODO: Do we need the clamp here? distSqr is already clamped, so NaN should not be possible (but +inf is).
    float denom = max(FLT_MIN, cosTheta * hit.triangleArea);
    return distSqr / denom;
#endif // !_NUM_MESH_LIGHTS
}
