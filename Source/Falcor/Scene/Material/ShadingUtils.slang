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
import Scene.ShadingData;
import Scene.SceneTypes;
import Scene.Material.MaterialTypes;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;

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
    n.xy = rg * 2 - 1;

    // Saturate because error from BC5 can break the sqrt
    n.z = saturate(dot(rg, rg)); // z = r*r + g*g
    n.z = sqrt(1 - n.z);
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
    \param[in] tangent Tangent in world space (xyz) and bitangent sign (w). The tangent is *only* valid when w != 0.
*/
void applyNormalMap(inout ShadingData sd, const NormalMapType type, const float3 encodedNormal, const float4 tangentW)
{
    float3 mapN = {};
    switch (type)
    {
    case NormalMapType::RGB:
        mapN = rgbToNormal(encodedNormal);
        break;
    case NormalMapType::RG:
        mapN = rgToNormal(encodedNormal.rg);
        break;
    default:
        return;
    }

    // Note if the normal ends up being parallel to the tangent, the tangent frame cannot be orthonormalized.
    // That case is rare enough that it is probably not worth the runtime cost to check for it here.
    // If it occurs we should foremost fix the asset, or if problems persist add a check here.

    // Apply the transformation.
    sd.N = normalize(sd.T * mapN.x + sd.B * mapN.y + sd.N * mapN.z);
    sd.T = normalize(tangentW.xyz - sd.N * dot(tangentW.xyz, sd.N));
    sd.B = cross(sd.N, sd.T) * tangentW.w;
}

/** Computes an orthonormal tangent space based on the normal and given tangent.
    \param[in,out] sd ShadingData struct that is updated.
    \param[in] tangent Interpolated tangent in world space (xyz) and bitangent sign (w). The tangent is *only* valid when w is != 0.
    \return True if a valid tangent space was computed based on the supplied tangent.
*/
bool computeTangentSpace(inout ShadingData sd, const float4 tangentW)
{
    // Check that tangent space exists and can be safely orthonormalized.
    // Otherwise invent a tanget frame based on the normal.
    // We check that:
    //  - Tangent exists, this is indicated by a nonzero sign (w).
    //  - It has nonzero length. Zeros can occur due to interpolation or bad assets.
    //  - It is not parallel to the normal. This can occur due to normal mapping or bad assets.
    //  - It does not have NaNs. These will propagate and trigger the fallback.

    float NdotT = dot(tangentW.xyz, sd.N);
    bool nonParallel = abs(NdotT) < 0.9999f;
    bool nonZero = dot(tangentW.xyz, tangentW.xyz) > 0.f;

    bool valid = tangentW.w != 0.f && nonZero && nonParallel;
    if (valid)
    {
        sd.T = normalize(tangentW.xyz - sd.N * NdotT);
        sd.B = cross(sd.N, sd.T) * tangentW.w;
    }
    else
    {
        sd.T = perp_stark(sd.N);
        sd.B = cross(sd.N, sd.T);
    }

    return valid;
}

/** Helper function to adjust the shading normal to reduce black pixels due to back-facing view direction.
    Note: This breaks the reciprocity of the BSDF!
*/
void adjustShadingNormal(inout ShadingData sd, VertexData v)
{
    float3 Ng = sd.frontFacing ? v.faceNormalW : -v.faceNormalW;
    float3 Ns = sd.N;

    // Blend the shading normal towards the geometric normal at grazing angles.
    // This is to avoid the view vector from becoming back-facing.
    const float kCosThetaThreshold = 0.1f;
    float cosTheta = dot(sd.V, Ns);
    if (cosTheta <= kCosThetaThreshold)
    {
        float t = saturate(cosTheta * (1.f / kCosThetaThreshold));
        sd.N = normalize(lerp(Ng, Ns, t));
        computeTangentSpace(sd, v.tangentW);
    }
}
