/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
import Rendering.Materials.IBSDF;
import Scene.ShadingData;
import Scene.Material.MaterialTypes;
import Utils.Color.ColorHelpers;
__exported import Utils.Math.ShadingFrame;

// If set, the backfacing side in the hemisphere checks is forced to black.
#define FALCOR_BACKFACE_BLACK 1

/** Convert RGB to normal (unnormalized).
*/
float3 rgbToNormal(float3 rgb)
{
    return rgb * 2.f - 1.f;
}

/** Convert RG to normal (unnormalized).
*/
float3 rgToNormal(float2 rg)
{
    float3 n;
    n.xy = rg * 2.f - 1.f;

    // Saturate because error from BC5 can break the sqrt
    n.z = saturate(dot(rg, rg)); // z = r*r + g*g
    n.z = sqrt(1.f - n.z);
    return n;
}

// TODO: this function is broken an may return negative values.
[Differentiable]
float getMetallic(float3 diffuse, float3 spec)
{
    // This is based on the way that UE4 and Substance Painter 2 converts base+metallic+specular level to diffuse/spec colors
    // We don't have the specular level information, so the assumption is that it is equal to 0.5 (based on the UE4 documentation)
    // Note that I'm using the luminance here instead of the actual colors. The reason is that there's no guaraentee that all RGB channels will end up with the same metallic value
    float d = luminance(diffuse);
    float s = luminance(spec);
    if (s == 0) return 0;
    float a = 0.04;
    float b = s + d - 0.08;
    float c = 0.04 - s;
    float root = sqrt(b*b - 0.16*c);
    float m = (root - b) * 12.5;
    return m;
}

/** Compute shading frame at shading point by applying normal mapping.
    This function perturbs the shading normal using a local normal sampled from a normal map.
    \param[in] sd Shading data.
    \param[in] type Normal map encoding.
    \param[in] encodedNormal Encoded normal.
    \return Shading frame in world space.
*/
ShadingFrame computeShadingFrameFromNormalMap(const ShadingData sd, const NormalMapType type, const float3 encodedNormal)
{
    // Compute perturbed normal in the local frame with the +z axis up.
    float3 mapN = { 0.f, 0.f, 1.f };
    switch (type)
    {
    case NormalMapType::RGB:
        mapN = rgbToNormal(encodedNormal);
        break;
    case NormalMapType::RG:
        mapN = rgToNormal(encodedNormal.rg);
        break;
    case NormalMapType::None:
        return sd.frame;
    }

    // Note if the normal ends up being parallel to the tangent, the tangent frame cannot be orthonormalized.
    // That case is rare enough that it is probably not worth the runtime cost to check for it here.
    // If it occurs we should foremost fix the asset, or if problems persist add a check here.

    // Apply the transformation.
    return ShadingFrame(sd.frame.fromLocal(mapN), sd.tangentW);
}

/** Flip shading normal for backfacing hit on double-sided material.
    \param[in] sd Shading data.
    \param[in,out] sf Shading frame that will be modified.
*/
[Differentiable]
void flipShadingNormal(const ShadingData sd, inout ShadingFrame sf)
{
    if (!sd.frontFacing && sd.mtl.isDoubleSided())
    {
        sf.N = -sf.N;
    }
}

/** Adjusts the normal of the supplied shading frame to reduce black pixels due to back-facing view direction.
    Note: This breaks the reciprocity of the BSDF!
    \param[in] sd Shading data.
    \param[in,out] sf Shading frame that will be adjusted.
*/
void adjustShadingNormal(const ShadingData sd, inout ShadingFrame sf)
{
    // Note: sd.V and Ng below always lie on the same side (as the front-facing flag is computed based on sd.V).
    // The shading normal sf.N may lie on either side depending on whether we're shading the front or back.
    // We orient Ns to lie on the same side as sd.V and Ng for the computations below.
    // The final adjusted normal is oriented to lie on the same side as the original shading normal.
    float3 Ng = sd.getOrientedFaceNormal();
    float signN = dot(sf.N, Ng) >= 0.f ? 1.f : -1.f;
    float3 Ns = signN * sf.N;

    // Blend the shading normal towards the geometric normal at grazing angles.
    // This is to avoid the view vector from becoming back-facing.
    const float kCosThetaThreshold = 0.1f;
    float cosTheta = dot(sd.V, Ns);
    if (cosTheta <= kCosThetaThreshold)
    {
        float t = saturate(cosTheta * (1.f / kCosThetaThreshold));
        sf = ShadingFrame(signN * lerp(Ng, Ns, t), sd.tangentW);
    }
}

/** Hemisphere check for reflection.
    This checks that both incident and outgoing directions are on the same geometric side,
    and that they are both in the upper hemisphere in the local frame.
    \param[in] sd Shading data.
    \param[in] sf Local shading frame.
    \param[in] wiLocal Incident direction in local space.
    \param[in] woLocal Outgoing direction in local space.
    \param[in] wo Outgoing direction in world space.
    \return True if wi/wo represent valid vectors for reflection.
*/
bool isValidHemisphereReflection(const ShadingData sd, const ShadingFrame sf, const float3 wiLocal, const float3 woLocal, const float3 wo)
{
    // Check that wi/wo are in the upper hemisphere around the shading normal.
    if (min(wiLocal.z, woLocal.z) < kMinCosTheta) return false;

    // Check that wi/wo are on the same geometric side.
    bool wiTop = sd.frontFacing; // The flag is computed dot(wi, faceN) >= 0.f.
    bool woTop = dot(wo, sd.faceN) >= 0.f;
    if (wiTop != woTop) return false;

#if FALCOR_BACKFACE_BLACK
    // Additionally check that we're on the same geometric side as the shading normal.
    bool shadingTop = dot(sf.N, sd.faceN) >= 0.f;
    if (wiTop != shadingTop) return false;
#endif

    return true;
}

/** Hemisphere check for transmission.
    This checks that the incident and outgoing directions are on opposite geometric sides,
    and that they are in the upper/lower hemisphere in the local frame.
    \param[in] sd Shading data.
    \param[in] sf Local shading frame.
    \param[in] wiLocal Incident direction in local space.
    \param[in] woLocal Outgoing direction in local space.
    \param[in] wo Outgoing direction in world space.
    \return True if wi/wo represent valid vectors for transmission.
*/
bool isValidHemisphereTransmission(const ShadingData sd, const ShadingFrame sf, const float3 wiLocal, const float3 woLocal, const float3 wo)
{
    // Check that wi/wo are in the upper/lower hemisphere around the shading normal.
    if (min(wiLocal.z, -woLocal.z) < kMinCosTheta) return false;

    // Check that wi/wo are on opposite geometric sides.
    bool wiTop = sd.frontFacing; // The flag is computed dot(wi, faceN) >= 0.f.
    bool woTop = dot(wo, sd.faceN) >= 0.f;
    if (wiTop == woTop) return false;

#if FALCOR_BACKFACE_BLACK
    // Additionally check that we're on the same geometric side as the shading normal.
    bool shadingTop = dot(sf.N, sd.faceN) >= 0.f;
    if (wiTop != shadingTop) return false;
#endif

    return true;
}

/** Hemisphere check for reflection or transmission.
    This check ensures that both incident and outgoing directions are on the correct geometric sides,
    and that they are in the correct hemispheres in the local frame.
    \param[in] sd Shading data.
    \param[in] sf Local shading frame.
    \param[in] wiLocal Incident direction in local space.
    \param[in] woLocal Outgoing direction in local space.
    \param[in] wo Outgoing direction in world space.
    \param[in] fullSphere If true allow the full sphere of directions, otherwise wi in the lower hemisphere is rejected and back-faces are black.
    \return True if wi/wo represent valid vectors for reflection or transmission.
*/
bool isValidHemisphereReflectionOrTransmission(const ShadingData sd, const ShadingFrame sf, const float3 wiLocal, const float3 woLocal, const float3 wo, const bool fullSphere = false)
{
    const bool isTransmission = wiLocal.z * woLocal.z < 0.f;

    // Check that wi/wo are in the upper/lower hemisphere around the shading normal.
    float cosThetaI = fullSphere ? abs(wiLocal.z) : wiLocal.z;
    if (min(cosThetaI, abs(woLocal.z)) < kMinCosTheta) return false;

    // Check that wi/wo are on the same geometric side.
    bool wiTop = sd.frontFacing; // The flag is computed dot(wi, faceN) >= 0.f.
    bool woTop = dot(wo, sd.faceN) >= 0.f;

    if (isTransmission)
    {
        if (wiTop == woTop) return false;
    }
    else // reflection
    {
        if (wiTop != woTop) return false;
    }

#if FALCOR_BACKFACE_BLACK
    if (!fullSphere)
    {
        // Additionally check that we're on the same geometric side as the shading normal.
        bool shadingTop = dot(sf.N, sd.faceN) >= 0.f;
        if (wiTop != shadingTop) return false;
    }
#endif

    return true;
}
