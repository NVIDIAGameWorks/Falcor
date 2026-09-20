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
#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Utils.Math.MathHelpers;

#ifndef LIGHT_SAMPLE_ENV_INDEX_BITS
#error "LightSample bit allocations not defined!"
#endif

enum class LightKind
{
    Env = (0 << 30),
    Emissive = (1 << 30),
    Analytic = (2 << 30),
};

typedef uint PackedLightSample;

namespace LightTileResampling
{
struct LightSample
{
    static const uint kInvalid = 0xffffffff;

    static const uint kKindBits = 2;
    static const uint kKindOffset = 30;
    static const uint kKindMask = ((1 << kKindBits) - 1) << kKindOffset;

    static const uint kPositionIndexMask = ~kKindMask;

    static const uint kEnvIndexBits = LIGHT_SAMPLE_ENV_INDEX_BITS;
    static const uint kEnvIndexMask = (1 << kEnvIndexBits) - 1;

    static const uint kEnvPositionBits = LIGHT_SAMPLE_ENV_POSITION_BITS;
    static const uint kEnvPositionOffset = kEnvIndexBits;
    static const uint kEnvPositionMask = ((1 << kEnvPositionBits) - 1) << kEnvPositionOffset;

    static const uint kEmissiveIndexBits = LIGHT_SAMPLE_EMISSIVE_INDEX_BITS;
    static const uint kEmissiveIndexMask = (1 << kEmissiveIndexBits) - 1;

    static const uint kEmissivePositionBits = LIGHT_SAMPLE_EMISSIVE_POSITION_BITS;
    static const uint kEmissivePositionOffset = kEmissiveIndexBits;
    static const uint kEmissivePositionMask = ((1 << kEmissivePositionBits) - 1) << kEmissivePositionOffset;

    static const uint kAnalyticIndexBits = LIGHT_SAMPLE_ANALYTIC_INDEX_BITS;
    static const uint kAnalyticIndexMask = (1 << kAnalyticIndexBits) - 1;

    static const uint kAnalyticPositionBits = LIGHT_SAMPLE_ANALYTIC_POSITION_BITS;
    static const uint kAnalyticPositionOffset = kAnalyticIndexBits;
    static const uint kAnalyticPositionMask = ((1 << kAnalyticPositionBits) - 1) << kAnalyticPositionOffset;

    static uint encodePosition(const float2 position, const uint bitsPerComponent)
    {
        const uint mask = (1 << bitsPerComponent) - 1;
        uint2 tmp = uint2(position * float(1 << bitsPerComponent)) & mask;
        return tmp.x | (tmp.y << bitsPerComponent);
    }

    static float2 decodePosition(const uint position, const uint bitsPerComponent)
    {
        const uint mask = (1 << bitsPerComponent) - 1;
        return (uint2(position, position >> bitsPerComponent) & mask) / float(1 << bitsPerComponent);
    }

    uint data;

    static LightSample createInvalid()
    {
        LightSample ls;
        ls.data = kInvalid;
        return ls;
    }

    static LightSample createEnv(uint index, uint random)
    {
        LightSample ls;
        ls.data = uint(LightKind::Env) | (index & kEnvIndexMask) | (random & kEnvPositionMask);
        return ls;
    }

    static LightSample createEnv(const float3 dir)
    {
        uint2 dim = gScene.envMap.getDimensions();
        float2 uv = gScene.envMap.worldToUv(dir);
        float2 texel = uv * dim;
        uint index = uint(texel.y) * dim.x + uint(texel.x);
        uint position = encodePosition(frac(texel), kEnvPositionBits / 2);

        LightSample ls;
        ls.data = uint(LightKind::Env) | (index & kEnvIndexMask) | (position << kEnvPositionOffset);
        return ls;
    }

    static LightSample createEmissive(uint index, uint random)
    {
        LightSample ls;
        ls.data = uint(LightKind::Emissive) | (index & kEmissiveIndexMask) | (random & kEmissivePositionMask);
        return ls;
    }

    static LightSample createEmissive(uint index, float2 uv)
    {
        uint position = encodePosition(uv, kEmissivePositionBits / 2);

        LightSample ls = {};
        ls.data = uint(LightKind::Emissive) | (index & kEmissiveIndexMask) | (position << kEmissivePositionOffset);
        return ls;
    }

    static LightSample createAnalytic(uint index, uint random)
    {
        LightSample ls;
        ls.data = uint(LightKind::Analytic) | (index & kAnalyticIndexMask) | (random & kAnalyticPositionMask);
        return ls;
    }

    static LightSample createFromPacked(const PackedLightSample packed)
    {
        LightSample ls;
        ls.unpack(packed);
        return ls;
    }

    PackedLightSample pack() { return PackedLightSample(data); }

    [mutating]
    void unpack(const PackedLightSample packed) { data = uint(packed); }

    bool isValid() { return data != kInvalid; }

    LightKind getKind() { return LightKind(data & kKindMask); }

    uint getEnvIndex() { return data & kEnvIndexMask; }
    uint getEmissiveIndex() { return data & kEmissiveIndexMask; }
    uint getAnalyticIndex() { return data & kAnalyticIndexMask; }

    uint getIndex()
    {
        switch (getKind())
        {
        case LightKind::Env:
            return getEnvIndex();
        case LightKind::Emissive:
            return getEmissiveIndex();
        case LightKind::Analytic:
            return getAnalyticIndex();
        }
        return 0;
    }

    float2 getEnvPosition() { return decodePosition(data >> kEnvPositionOffset, kEnvPositionBits / 2); }
    float2 getEmissivePosition() { return decodePosition(data >> kEmissivePositionOffset, kEmissivePositionBits / 2); }
    float2 getAnalyticPosition() { return decodePosition(data >> kAnalyticPositionOffset, kAnalyticPositionBits / 2); }

    float3 getEnvWorldDir()
    {
        uint texel = getEnvIndex();
        uint2 dim = gScene.envMap.getDimensions();
        float2 uv = (float2(texel % dim.x, texel / dim.x) + getEnvPosition()) / dim;
        return gScene.envMap.toWorld(latlong_map_to_world(uv));
    }

    float2 getEmissiveUV()
    {
        float2 uv = getEmissivePosition();
        return uv.x + uv.y > 1.f ? 1.f - uv : uv;
    }

    float3 getEmissiveBarycentrics()
    {
        float2 uv = getEmissiveUV();
        return float3(1.f - uv.x - uv.y, uv.x, uv.y);
    }
}
}
