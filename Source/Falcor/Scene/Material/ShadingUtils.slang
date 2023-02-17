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
import Scene.SceneTypes;
import Scene.Material.MaterialTypes;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;

// If set, the backfacing side in the hemisphere checks is forced to black.
#define FALCOR_BACKFACE_BLACK 1

/** Shading frame in world space.
    The vectors TBN form an orthonormal basis.
*/
struct ShadingFrame
{
    float3 T;   ///< Shading tangent. Normalized.
    float3 B;   ///< Shading bitangent. Normalized.
    float3 N;   ///< Shading normal. Normalized.

    __init() {}

    __init(const ShadingData sd)
    {
        this.T = sd.T;
        this.B = sd.B;
        this.N = sd.N;
    }

    /** Initialize shading frame from normal and tangent.
        It is assumed the shading frame can be safely orthonormalized. If in doubt, use `createSafe` instead.
        \param[in] Normal in world space. Not normalized.
        \param[in] Target tangent in world space (xyz) and handedness sign (w). Not normalized.
    */
    __init(const float3 normalW, const float4 tangentW)
    {
        this.N = normalize(normalW);
        orthonormalize(tangentW);
    }

    /** Create a shading frame from normal and tangent.
        If the shading frame cannot be safely orthonormalized, a new tangent is invented based on the normal.
        \param[in] Normal in world space. Not normalized.
        \param[in] Target tangent in world space (xyz) and handedness sign (w). Not normalized.
        \return New shading frame.
    */
    static ShadingFrame createSafe(const float3 normalW, const float4 tangentW)
    {
        ShadingFrame sf;
        sf.N = normalize(normalW);
        sf.orthonormalizeSafe(tangentW);
        return sf;
    }

    /** Transform vector from the local shading frame to world space.
        \param[in] v Vector in local space.
        \return Vector in world space.
    */
    float3 fromLocal(const float3 v)
    {
        return T * v.x + B * v.y + N * v.z;
    }

    /** Transform vector from world space to the local shading frame.
        \param[in] v Vector in world space.
        \return Vector in local space.
    */
    float3 toLocal(const float3 v)
    {
        return float3(dot(v, T), dot(v, B), dot(v, N));
    }

    /** Returns sign that gets applied to such that B = cross(N, T) * getHandednessSign();
        \return +1 for right handed, and -1 for left handed frames.
    */
    float getHandednessSign()
    {
        return dot(cross(N, T), B) >= 0.f ? 1.f : -1.f;
    }

    /** Orthonormalize the shading frame.
        Vectors T and B are adjusted to make an orthonormal frame where T lies in the same plane as N and the supplied target tangent.
        It is assumed that the tangent is nonzero and not parallel to the normal, and that the sign (w) is +-1.0.
        \param[in] tangentW Target tangent in world space (xyz) and handedness sign (w). Not normalized.
    */
    [mutating]
    void orthonormalize(const float4 tangentW)
    {
        T = normalize(tangentW.xyz - N * dot(tangentW.xyz, N));
        B = cross(N, T) * tangentW.w;
    }

    /** Orthonormalize the shading frame.
        Vectors T and B are adjusted to make an orthonormal frame where T lies in the same plane as N and the supplied target tangent.
        If the shading frame cannot be safely orthonormalized, a new tangent is invented based on the normal.
        \param[in] tangentW Target tangent in world space (xyz) and handedness sign (w). Not normalized.
        \return True if a valid tangent space was computed based on the supplied tangent, or false if a tangent space was invented.
    */
    [mutating]
    bool orthonormalizeSafe(const float4 tangentW)
    {
        // Check that tangent space exists and can be safely orthonormalized.
        // Otherwise invent a tanget frame based on the normal.
        // We check that:
        //  - Tangent exists, this is indicated by a sign (w) that is +-1.0.
        //  - It has nonzero length. Zeros can occur due to interpolation or bad assets.
        //  - It is not parallel to the normal. This can occur due to normal mapping or bad assets.
        //  - It does not have NaNs. These will propagate and trigger the fallback.

        float NdotT = dot(tangentW.xyz, N);
        bool nonParallel = abs(NdotT) < 0.9999f;
        bool nonZero = dot(tangentW.xyz, tangentW.xyz) > 0.f;
        bool valid = abs(tangentW.w) == 1.0f && nonZero && nonParallel;

        if (valid)
        {
            T = normalize(tangentW.xyz - N * NdotT);
            B = cross(N, T) * tangentW.w;
        }
        else
        {
            T = perp_stark(N);
            B = cross(N, T);
        }

        return valid;
    }
};

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

/** Apply normal map.
    This function perturbs the shading normal using a local normal sampled from a normal map.
    \param[in,out] sd ShadingData struct that is updated.
    \param[in] type Normal map type.
    \param[in] encodedNormal Encoded normal loaded from normal map.
*/
void applyNormalMap(inout ShadingData sd, const NormalMapType type, const float3 encodedNormal)
{
    ShadingFrame sf = computeShadingFrameFromNormalMap(sd, type, encodedNormal);
    sd.T = sf.T;
    sd.B = sf.B;
    sd.N = sf.N;
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
        return ShadingFrame(sd);
    }

    // Note if the normal ends up being parallel to the tangent, the tangent frame cannot be orthonormalized.
    // That case is rare enough that it is probably not worth the runtime cost to check for it here.
    // If it occurs we should foremost fix the asset, or if problems persist add a check here.

    // Apply the transformation.
    return ShadingFrame(sd.T * mapN.x + sd.B * mapN.y + sd.N * mapN.z, sd.tangentW);
}

/** Computes an orthonormal tangent space based on the normal and given tangent.
    \param[in,out] sd ShadingData struct that is updated.
    \param[in] tangent Interpolated tangent in world space (xyz) and bitangent sign (w). The tangent is *only* valid when w is +-1.0.
    \return True if a valid tangent space was computed based on the supplied tangent, or false if a tangent space was invented.
*/
bool computeTangentSpace(inout ShadingData sd, const float4 tangentW)
{
    ShadingFrame sf = ShadingFrame(sd);
    bool valid = sf.orthonormalizeSafe(tangentW);
    sd.T = sf.T;
    sd.B = sf.B;
    sd.N = sf.N;
    return valid;
}

/** Helper function to adjust the shading normal to reduce black pixels due to back-facing view direction.
    Note: This breaks the reciprocity of the BSDF!
*/
void adjustShadingNormal(inout ShadingData sd)
{
    ShadingFrame sf = ShadingFrame(sd);
    adjustShadingNormal(sd, sf);
    sd.T = sf.T;
    sd.B = sf.B;
    sd.N = sf.N;
}

/** Adjusts the normal of the supplied shading frame to reduce black pixels due to back-facing view direction.
    Note: This breaks the reciprocity of the BSDF!
    \param[in] sd Shading data.
    \param[in,out] sf Shading frame that will be adjusted.
*/
void adjustShadingNormal(const ShadingData sd, inout ShadingFrame sf)
{
    float3 Ng = sd.frontFacing ? sd.faceN : -sd.faceN;
    float3 Ns = sf.N;

    // Blend the shading normal towards the geometric normal at grazing angles.
    // This is to avoid the view vector from becoming back-facing.
    const float kCosThetaThreshold = 0.1f;
    float cosTheta = dot(sd.V, Ns);
    if (cosTheta <= kCosThetaThreshold)
    {
        float t = saturate(cosTheta * (1.f / kCosThetaThreshold));
        sf = ShadingFrame(lerp(Ng, Ns, t), sd.tangentW);
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
    \return True if wi/wo represent valid vectors for reflection or transmission.
*/
bool isValidHemisphereReflectionOrTransmission(const ShadingData sd, const ShadingFrame sf, const float3 wiLocal, const float3 woLocal, const float3 wo)
{
    const bool isTransmission = woLocal.z < 0.f;

    // Check that wi/wo are in the upper/lower hemisphere around the shading normal.
    if (min(wiLocal.z, abs(woLocal.z)) < kMinCosTheta) return false;

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
    // Additionally check that we're on the same geometric side as the shading normal.
    bool shadingTop = dot(sf.N, sd.faceN) >= 0.f;
    if (wiTop != shadingTop) return false;
#endif

    return true;
}
