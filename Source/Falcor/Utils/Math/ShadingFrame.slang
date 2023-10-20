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
import Utils.Math.MathHelpers;

/**
 * Shading frame in world space.
 * The vectors TBN form an orthonormal basis.
 */
struct ShadingFrame : IDifferentiable
{
    float3 T; ///< Shading tangent. Normalized.
    float3 B; ///< Shading bitangent. Normalized.
    float3 N; ///< Shading normal. Normalized.

    /**
     * Initialize shading frame from normal and tangent.
     * It is assumed the shading frame can be safely orthonormalized. If in doubt, use `createSafe` instead.
     * @param[in] Normal in world space. Not normalized.
     * @param[in] Target tangent in world space (xyz) and handedness sign (w). Not normalized.
     */
    [Differentiable]
    __init(const float3 normalW, const float4 tangentW)
    {
        this.N = normalize(normalW);
        orthonormalize(tangentW);
    }

    /**
     * Create identity shading frame.
     * @return New shading frame.
     */
    [Differentiable]
    static ShadingFrame createIdentity()
    {
        ShadingFrame sf;
        sf.T = float3(1.0f, 0.0f, 0.0f);
        sf.B = float3(0.0f, 1.0f, 0.0f);
        sf.N = float3(0.0f, 0.0f, 1.0f);
        return sf;
    }

    /**
     * Create a shading frame from normal and tangent.
     * If the shading frame cannot be safely orthonormalized, a new tangent is invented based on the normal.
     * @param[in] normalW Normal in world space. Not normalized.
     * @param[in] tangentW Target tangent in world space (xyz) and handedness sign (w). Not normalized.
     * @param[out] valid True if a valid tangent space was computed based on the supplied tangent, or false if a tangent space was invented.
     * @return New shading frame.
     */
    [Differentiable]
    static ShadingFrame createSafe(const float3 normalW, const float4 tangentW, out bool valid)
    {
        ShadingFrame sf;
        sf.N = normalize(normalW);
        valid = sf.orthonormalizeSafe(tangentW);
        return sf;
    }

    /**
     * Transform vector from the local shading frame to world space.
     * @param[in] v Vector in local space.
     * @return Vector in world space.
     */
    [Differentiable]
    float3 fromLocal(const float3 v) { return T * v.x + B * v.y + N * v.z; }

    /**
     * Transform vector from world space to the local shading frame.
     * @param[in] v Vector in world space.
     * @return Vector in local space.
     */
    [Differentiable]
    float3 toLocal(const float3 v) { return float3(dot(v, T), dot(v, B), dot(v, N)); }

    /**
     * Returns sign that gets applied to such that B = cross(N, T) * getHandednessSign();
     * @return +1 for right handed, and -1 for left handed frames.
     */
    float getHandednessSign() { return dot(cross(N, T), B) >= 0.f ? 1.f : -1.f; }

    /**
     * Orthonormalize the shading frame.
     * Vectors T and B are adjusted to make an orthonormal frame where T lies in the same plane as N and the supplied target tangent.
     * It is assumed that the tangent is nonzero and not parallel to the normal, and that the sign (w) is +-1.0.
     * @param[in] tangentW Target tangent in world space (xyz) and handedness sign (w). Not normalized.
     */
    [mutating]
    [Differentiable]
    void orthonormalize(const float4 tangentW)
    {
        T = normalize(tangentW.xyz - N * dot(tangentW.xyz, N));
        B = cross(N, T) * tangentW.w;
    }

    /**
     * Orthonormalize the shading frame.
     * Vectors T and B are adjusted to make an orthonormal frame where T lies in the same plane as N and the supplied target tangent.
     * If the shading frame cannot be safely orthonormalized, a new tangent is invented based on the normal.
     * @param[in] tangentW Target tangent in world space (xyz) and handedness sign (w). Not normalized.
     * @return True if a valid tangent space was computed based on the supplied tangent, or false if a tangent space was invented.
     */
    [mutating]
    [Differentiable]
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
            T = no_diff perp_stark(N);
            B = cross(N, T);
        }

        return valid;
    }
};
