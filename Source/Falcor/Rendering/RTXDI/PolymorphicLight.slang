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
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Rendering.Lights.LightHelpers;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
__exported import PackedTypes;

/** Polymorphic light type.
*/
enum class PolymorphicLightType
{
    Emissive,   ///< Emissive triangle lights.
    Analytic,   ///< Analytic lights (both local and infinite).
    Env,        ///< Environment light.
};

/** Extend PackedPolymorphicLight with helpers to pack/unpack the header.
*/
extension PackedPolymorphicLight
{
    static const uint kTypeBits = 2;
    static const uint kTypeOffset = 32 - kTypeBits;
    static const uint kExtraMask = (1 << kTypeOffset) - 1;

    /** Set the header.
        \param[in] type Light type.
        \param[in] extra Extra bits.
    */
    [mutating] void setHeader(const PolymorphicLightType type, const uint extra = 0)
    {
        header = (uint(type) << kTypeOffset) | (extra & kExtraMask);
    }

    /** Get the light type.
        \return The light type.
    */
    PolymorphicLightType getType()
    {
        return PolymorphicLightType(header >> kTypeOffset);
    }

    /** Get the extra header bits.
        \return The extra bits.
    */
    uint getExtra()
    {
        return header & kExtraMask;
    }
};

/** Stores a light sample taken from a polymorphic light.
*/
struct PolymorphicLightSample
{
    float3 position;    ///< Position on the light in world-space.
    float3 normal;      ///< Normal on the light in world-space.
    float3 direction;   ///< Direction from the shading point to the light in world-space.
    float distance;     ///< Distance from the shading point to the light in world-space.
    float3 Li;          ///< Incident radiance at the shading point (unshadowed), pre-divided by the PDF.
    float pdf;          ///< PDF wrt. solid angle.
    uint flags;         ///< Light type enum.

    property PolymorphicLightType lightType
    {
        get() { return (PolymorphicLightType)flags; }
        set() { flags = (uint)newValue; }
    }

    // True if sample is valid.
    property bool valid
    {
        get() { return pdf > 0.f; }
    }
};

/** Interface for polymorphic lights.
*/
interface IPolymorphicLight
{
    /** Pack data into a common packed type.
        \return Packed polymorphic light.
    */
    PackedPolymorphicLight pack();

    /** Compute the total flux of the light.
        Note: Infinite/distant lights must return 0.
        \return Total flux.
    */
    float getFlux();

#ifdef RTXDI_POLYMORPHIC_LIGHT_SAMPLING
    /** Compute a light sample.
        \param[in] shadingPosition Shading position.
        \param[in] u Uniform random number.
        \param[in] scalarLi Compute scalar radiance instead of RGB radiance.
        \return The light sample.
    */
    PolymorphicLightSample calcSample(const float3 shadingPosition, const float2 u, const bool scalarLi);
#endif // RTXDI_POLYMORPHIC_LIGHT_SAMPLING
};

/** Represents emissive triangle lights.
*/
struct EmissiveLight : IPolymorphicLight
{
    uint triIdx;            ///< Global triangle index in the scene's LightCollection.
    float3 position;        ///< Position in world-space.
    float3 edge0;           ///< First triangle edge in world-space.
    float3 edge1;           ///< Second triangle edge in world-space.
    float averageRadiance;  ///< Average emitted radiance (scalar).

    __init(const PackedPolymorphicLight packed)
    {
        triIdx = packed.getExtra();
        position = asfloat(packed.data0);
        edge0 = f16tof32(packed.data1.xyz & 0xffff);
        edge1 = f16tof32(packed.data1.xyz >> 16);
        averageRadiance = asfloat(packed.data1.w);
    }

    PackedPolymorphicLight pack()
    {
        PackedPolymorphicLight packed = {};
        packed.setHeader(PolymorphicLightType::Emissive, triIdx);
        packed.data0 = asuint(position);
        packed.data1.xyz = (f32tof16(edge0) & 0xffff) | (f32tof16(edge1) << 16);
        packed.data1.w = asuint(averageRadiance);
        return packed;
    }

    float getArea()
    {
        return 0.5f * length(cross(edge0, edge1));
    }

    float getFlux()
    {
        // Note: This assumes emission to be diffuse. We currently take the same assumption in LightCollection.
        return getArea() * averageRadiance * M_PI;
    }

#ifdef RTXDI_POLYMORPHIC_LIGHT_SAMPLING
    PolymorphicLightSample calcSample(const float3 shadingPosition, const float2 u, const bool scalarLi)
    {
        PolymorphicLightSample ls = {};
        ls.lightType = PolymorphicLightType::Emissive;

        // Uniformly sample the triangle's area by folding the random numbers such that u & v sum to less than 1.
        const float2 uv = ((u.x + u.y) <= 1.f) ? u : 1.f - u;

        // Compute normal and area.
        float3 normal = cross(edge0, edge1);
        const float normalLength = length(normal);
        const float area = 0.5f * normalLength;
        normal /= normalLength;

        ls.position = position + uv.x * edge0 + uv.y * edge1;
        const float3 toLight = ls.position - shadingPosition; // Unnormalized light vector
        const float distSqr = max(FLT_MIN, dot(toLight, toLight)); // Clamp to avoid NaNs below
        ls.distance = sqrt(distSqr);
        ls.direction = toLight / ls.distance;
        ls.normal = normal;

        // Reject sample if triangle is back-facing.
        float cosTheta = dot(ls.normal, -ls.direction);
        if (cosTheta <= 0.f) return {};

        if (scalarLi)
        {
            ls.Li = averageRadiance;
        }
        else
        {
            if (gRTXDI.useEmissiveTextures)
            {
                // Load emissive triangle from scene.
                const EmissiveTriangle et = gScene.lightCollection.getTriangle(triIdx);

                // Evaluate texture coordinate on triangle.
                const float3 bary = { 1.f - uv.x - uv.y, uv.x, uv.y };
                const float2 texCoord = et.getTexCoord(bary);

                // Lookup emission from material.
                ls.Li = gScene.materials.evalEmissive(et.materialID, texCoord);
            }
            else
            {
                ls.Li = gScene.lightCollection.getAverageRadiance(triIdx);
            }
        }

        // Compute probability density with respect to solid angle from the shading point.
        // The farther away the light is and the larger the angle it is at, the larger the pdf becomes. The probability goes to infinity in the limit.
        // Note: Guard against div-by-zero here by clamping.
        float denom = max(FLT_MIN, cosTheta * area);
        ls.pdf = distSqr / denom;

        // Pre-divide incident radiance by the PDF.
        ls.Li *= denom / distSqr;

        return ls;
    }
#endif // RTXDI_POLYMORPHIC_LIGHT_SAMPLING
};

/** Represents analytic lights.
    For now these just store a reference to an analytic light in the global scene data.
*/
struct AnalyticLight : IPolymorphicLight
{
    uint lightID;

    __init(const PackedPolymorphicLight packed)
    {
        lightID = packed.getExtra();
    }

    PackedPolymorphicLight pack()
    {
        PackedPolymorphicLight packed = {};
        packed.setHeader(PolymorphicLightType::Analytic, lightID);
        return packed;
    }

    float getFlux()
    {
        const LightData light = gScene.getLight(lightID);

        switch (LightType(light.type))
        {
        case LightType::Point:
            return M_2PI * (1.f - light.cosOpeningAngle) * luminance(light.intensity);
        }

        return 0.f;
    }

#ifdef RTXDI_POLYMORPHIC_LIGHT_SAMPLING
    PolymorphicLightSample calcSample(const float3 shadingPosition, const float2 u, const bool scalarLi)
    {
        const LightData light = gScene.getLight(lightID);
        AnalyticLightSample als;
        bool valid = false;

        switch (LightType(light.type))
        {
        case LightType::Point:
            valid = samplePointLight(shadingPosition, light, als);
            als.pdf = 1.f;
            break;
        case LightType::Directional:
            valid = sampleDirectionalLight(shadingPosition, light, als);
            als.pdf = 1.f;
            break;
        case LightType::Distant:
            valid = sampleDistantLight(shadingPosition, light, u, als);
            break;
        }

        PolymorphicLightSample ls = {};
        ls.lightType = PolymorphicLightType::Analytic;
        if (!valid) return ls;

        ls.position = als.posW;
        ls.normal = als.normalW;
        ls.direction = als.dir;
        ls.distance = als.distance;
        ls.Li = scalarLi ? luminance(als.Li) : als.Li;
        ls.pdf = als.pdf;

        return ls;
    }
#endif // RTXDI_POLYMORPHIC_LIGHT_SAMPLING
}

/** Represents the environment light.
*/
struct EnvLight : IPolymorphicLight
{
    __init(const PackedPolymorphicLight packed)
    {
    }

    PackedPolymorphicLight pack()
    {
        PackedPolymorphicLight packed = {};
        packed.setHeader(PolymorphicLightType::Env);
        return packed;
    }

    float getFlux()
    {
        return 0.f;
    }

#ifdef RTXDI_POLYMORPHIC_LIGHT_SAMPLING
    PolymorphicLightSample calcSample(const float3 shadingPosition, const float2 u, const bool scalarLi)
    {
        PolymorphicLightSample ls = {};
        ls.lightType = PolymorphicLightType::Env;

        const float2 uv = u;
        float theta = (uv.y - 0.5f) * M_PI;
        float relativeSolidAngle = cos(theta);

        ls.direction = gScene.envMap.uvToWorld(uv);
        ls.normal = -ls.direction;
        ls.distance = FLT_MAX;
        ls.position = shadingPosition + ls.direction * ls.distance;
        if (scalarLi)
        {
            ls.Li = gRTXDI.envLightLuminanceTexture.SampleLevel(gScene.envMap.envSampler, uv, 0).rgb;
        }
        else
        {
            ls.Li = gScene.envMap.eval(uv);
        }
        ls.pdf = 1.f / max(FLT_MIN, 2.f * M_PI * M_PI * relativeSolidAngle);
        ls.Li /= ls.pdf;

        return ls;
    }
#endif // RTXDI_POLYMORPHIC_LIGHT_SAMPLING
};

/** Polymorphic light.
*/
struct PolymorphicLight : IPolymorphicLight
{
    PackedPolymorphicLight packed;

    __init(const PackedPolymorphicLight packed)
    {
        this.packed = packed;
    }

    PackedPolymorphicLight pack()
    {
        return packed;
    }

    PolymorphicLightType getType()
    {
        return packed.getType();
    }

    float getFlux()
    {
        switch (getType())
        {
        case PolymorphicLightType::Emissive:
            return EmissiveLight(packed).getFlux();
        case PolymorphicLightType::Analytic:
            return AnalyticLight(packed).getFlux();
        case PolymorphicLightType::Env:
            return EnvLight(packed).getFlux();
        }

        return 0.f;
    }

#ifdef RTXDI_POLYMORPHIC_LIGHT_SAMPLING
    PolymorphicLightSample calcSample(const float3 shadingPosition, const float2 u, const bool scalarLi)
    {
        switch (getType())
        {
        case PolymorphicLightType::Emissive:
            return EmissiveLight(packed).calcSample(shadingPosition, u, scalarLi);
        case PolymorphicLightType::Analytic:
            return AnalyticLight(packed).calcSample(shadingPosition, u, scalarLi);
        case PolymorphicLightType::Env:
            return EnvLight(packed).calcSample(shadingPosition, u, scalarLi);
        }

        return {};
    }
#endif // RTXDI_POLYMORPHIC_LIGHT_SAMPLING
};
