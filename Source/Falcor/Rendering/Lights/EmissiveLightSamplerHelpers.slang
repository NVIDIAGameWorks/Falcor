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

/** Shared utility functions for emissive light sampler implementations.

     These functions rely on LightCollection, which is a container holding the
     global list of all emissive triangles in the scene.

     TODO: It would've been nice if we could have these functions in the interface
     struct (representing the "base" class that "derived" classes can use).
 */

#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Scene.ShadingData;
__exported import Scene.Lights.LightCollection;
import Utils.Math.MathHelpers;
import Rendering.Lights.EmissiveLightSamplerInterface;

/** Samples an emissive triangle uniformly by area and evaluates the probability density function.
    Note that the triangle is only emitting on the front-facing side.
    \param[in] lights The light collection.
    \param[in] posW Shading point in world space.
    \param[in] triangleIndex Triangle index of sampled triangle.
    \param[in] u Uniform random number (2D).
    \param[out] ls Light sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool sampleTriangle(const float3 posW, const uint triangleIndex, const float2 u, out TriangleLightSample ls)
{
    ls = {};
    ls.triangleIndex = triangleIndex;
    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(triangleIndex);

    // Sample the triangle uniformly.
    const float3 barycentrics = sample_triangle(u);
    ls.posW = tri.getPosition(barycentrics);

    // Compute light vector and squared distance.
    float3 toLight = ls.posW - posW; // Unnormalized light vector
    const float distSqr = max(FLT_MIN, dot(toLight, toLight)); // Clamp to avoid NaNs below
    ls.distance = sqrt(distSqr);
    ls.dir = toLight / ls.distance; // Note: toLight can be zero.

    // Get triangle normal.
    ls.normalW = tri.normal;

    // Reject sample if triangle is back-facing.
    float cosTheta = dot(ls.normalW, -ls.dir);
    if (cosTheta <= 0.f) return false;

    // Evaluate emitted radiance.
    float2 uv = tri.getTexCoord(barycentrics);
    ls.Le = gScene.materials.evalEmissive(tri.materialID, uv);

    // DEMO21: Evaluate light profile.
    if (Scene::kUseLightProfile)
    {
        if (gScene.materials.getMaterialHeader(tri.materialID).isLightProfileEnabled())
        {
            ls.Le *= gScene.lightProfile.eval(dot(ls.normalW, -ls.dir));
        }
    }

    // Compute probability density with respect to solid angle from the shading point.
    // The farther away the light is and the larger the angle it is at, the larger the pdf becomes. The probability goes to infinity in the limit.
    // Note: Guard against div-by-zero here by clamping.
    float denom = max(FLT_MIN, cosTheta * tri.area);
    ls.pdf = distSqr / denom;

    // Save light sample triangle barycentric coordinates
    ls.uv = u;

    // TODO: We can simplify the expressions by using the unnormalized quantities for computing the pdf.
    //ls.pdf = -2.f * distSqr * ls.distance / (dot(N, L);    // Optimized (except N would have to be properly flipped above, before normalW is computed).
    return true;
}

/** Evaluates the PDF for a light sample given a hit point on an emissive triangle.
    \param[in] posW Shading point in world space.
    \param[in] hit Triangle hit data.
    \return Probability density with respect to solid angle at the shading point.
*/
float evalTrianglePdf(const float3 posW, const TriangleLightHit hit)
{
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
    float denom = max(FLT_MIN, cosTheta * gScene.lightCollection.getTriangle(hit.triangleIndex).area);
    return distSqr / denom;
}
