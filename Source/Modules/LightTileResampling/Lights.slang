/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MathHelpers;
import Utils.Math.PackedFormats;
import Utils.Color.ColorHelpers;
import Utils.Sampling.AliasTable;
__exported import Utils.Sampling.SampleGeneratorInterface;
__exported import LightSample;

struct EvaluatedLightSample
{
    float3 dir;
    float distance;
    float geomFactor;
    float emission;
    float pdf;
};

struct PackedLoadedLightSample
{
    uint lightSample;
    uint type;
    uint normal;
    uint emission;
    uint3 posOrDir;
    uint pdf;
};

using LightTileResampling;

struct LoadedLightSample
{
    enum class Type
    {
        Invalid,
        Area,
        AreaWithLightProfile,
        Point,
        Distant,
    };

    LightSample lightSample;
    Type type = Type::Invalid;
    float3 posOrDir;
    float3 normal;
    float emission;
    float pdf;

    EvaluatedLightSample eval(const float3 pos)
    {
        EvaluatedLightSample result = {};

        result.emission = emission;
        result.pdf = pdf;

        switch (type)
        {
        case Type::Area:
        case Type::AreaWithLightProfile:
        {
            float3 toLight = posOrDir - pos;
            float distSqr = max(FLT_MIN, dot(toLight, toLight));

            result.distance = sqrt(distSqr);
            result.dir = toLight / result.distance;
            result.geomFactor = max(0.f, dot(normal, -result.dir)) / distSqr;

            if (gScene.materials.lightProfilesEnabled())
            {
                if (type == Type::AreaWithLightProfile)
                {
                    result.emission *= gScene.materials.evalLightProfile(dot(normal, -result.dir));
                }
            }
        }
        break;
        case Type::Point:
        {
            float3 toLight = posOrDir - pos;
            float distSqr = max(FLT_MIN, dot(toLight, toLight));

            result.distance = sqrt(distSqr);
            result.dir = toLight / result.distance;
            result.geomFactor = 1.f / distSqr;
        }
        break;
        case Type::Distant:
        {
            result.distance = Lights::kRayTMax;
            result.dir = posOrDir;
            result.geomFactor = 1.f;
        }
        break;
        }

        return result;
    }

    static LoadedLightSample createFromPacked(const PackedLoadedLightSample packed)
    {
        LoadedLightSample lls;
        lls.unpack(packed);
        return lls;
    }

    PackedLoadedLightSample pack()
    {
        PackedLoadedLightSample packed;
        packed.lightSample = lightSample.pack();
        packed.type = uint(type);
        packed.normal = encodeNormal2x16(normal);
        packed.emission = asuint(emission);
        packed.posOrDir = asuint(posOrDir);
        packed.pdf = asuint(pdf);
        return packed;
    }

    [mutating]
    void unpack(const PackedLoadedLightSample packed)
    {
        lightSample.unpack(packed.lightSample);
        type = LoadedLightSample::Type(packed.type);
        normal = decodeNormal2x16(packed.normal);
        emission = asfloat(packed.emission);
        posOrDir = asfloat(packed.posOrDir);
        pdf = asfloat(packed.pdf);
    }
};

struct Lights
{
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;

    static const bool kUseEmissiveTextureForSampling = USE_EMISSIVE_TEXTURE_FOR_SAMPLING;
    static const bool kUseEmissiveTextureForShading = USE_EMISSIVE_TEXTURE_FOR_SHADING;

    static const float kRayTMax = FLT_MAX;
    static const float kRayEpsilon = 0.f;

    Buffer<float> envLightLuminance;

    AliasTable envLightAliasTable;
    AliasTable emissiveLightAliasTable;
    AliasTable analyticLightAliasTable;

    float envLightLuminanceFactor;

    float envLightSelectionProbability;
    float emissiveLightSelectionProbability;
    float analyticLightSelectionProbability;

    LightSample sampleEnv<S : ISampleGenerator>(float random, inout S sg)
    {
        return LightSample::createEnv(envLightAliasTable.sample(float2(random, sampleNext1D(sg))), sg.next());
    }

    LightSample sampleEmissive<S : ISampleGenerator>(float random, inout S sg)
    {
        return LightSample::createEmissive(emissiveLightAliasTable.sample(float2(random, sampleNext1D(sg))), sg.next());
    }

    LightSample sampleAnalytic<S : ISampleGenerator>(float random, inout S sg)
    {
        return LightSample::createAnalytic(analyticLightAliasTable.sample(float2(random, sampleNext1D(sg))), sg.next());
    }

    LoadedLightSample loadLightSample(const LightSample lightSample)
    {
        if (lightSample.isValid())
        {
            switch (lightSample.getKind())
            {
            case LightKind::Env:
                return loadEnvLightSample(lightSample);
            case LightKind::Emissive:
                return loadEmissiveLightSample(lightSample);
            case LightKind::Analytic:
                return loadAnalyticLightSample(lightSample);
            }
        }
        return { lightSample };
    }

    LoadedLightSample loadFromPacked(const PackedLightSample packed) { return loadLightSample(LightSample::createFromPacked(packed)); }

    LoadedLightSample loadEnvLightSample(const LightSample lightSample)
    {
        LoadedLightSample result = { lightSample, LoadedLightSample::Type::Distant };

        result.posOrDir = lightSample.getEnvWorldDir();
        float luminance = envLightLuminance[lightSample.getEnvIndex()];
        result.emission = luminance * envLightLuminanceFactor;
        result.pdf = envLightSelectionProbability * luminance / envLightAliasTable.weightSum;

        return result;
    }

    LoadedLightSample loadEmissiveLightSample(const LightSample lightSample)
    {
        LoadedLightSample result = { lightSample, LoadedLightSample::Type::Area };

        const EmissiveTriangle tri = gScene.lightCollection.getTriangle(lightSample.getEmissiveIndex());
        result.posOrDir = computeRayOrigin(tri.getPosition(lightSample.getEmissiveBarycentrics()), tri.normal);
        result.normal = tri.normal;
        float radiance = luminance(gScene.lightCollection.getAverageRadiance(lightSample.getEmissiveIndex()));
        result.emission = radiance;
        result.pdf = emissiveLightSelectionProbability * radiance / emissiveLightAliasTable.weightSum;

        if (kUseEmissiveTextureForSampling)
        {
            float2 texCoord = tri.getTexCoord(lightSample.getEmissiveBarycentrics());
            result.emission = luminance(gScene.materials.evalEmissive(tri.materialID, texCoord));
        }

        if (gScene.materials.hasLightProfile(tri.materialID))
        {
            result.type = LoadedLightSample::Type::AreaWithLightProfile;
        }

        return result;
    }

    LoadedLightSample loadAnalyticLightSample(const LightSample lightSample)
    {
        LoadedLightSample result = { lightSample };

        LightData light = gScene.getLight(lightSample.getAnalyticIndex());
        LightType type = LightType(light.type);
        if (type == LightType::Point)
        {
            result.type = LoadedLightSample::Type::Point;
            result.posOrDir = light.posW;
            result.emission = luminance(light.intensity);
            result.pdf = analyticLightSelectionProbability / analyticLightAliasTable.weightSum;
        }
        else if (type == LightType::Directional)
        {
            result.type = LoadedLightSample::Type::Distant;
            result.posOrDir = -light.dirW;
            result.emission = luminance(light.intensity);
            result.pdf = analyticLightSelectionProbability / analyticLightAliasTable.weightSum;
        }
        else if (type == LightType::Distant)
        {
            result.type = LoadedLightSample::Type::Distant;
            float3 dir = sample_cone(lightSample.getAnalyticPosition(), light.cosSubtendedAngle);
            result.posOrDir = normalize(mul((float3x3)light.transMat, dir));
            result.emission = luminance(light.intensity);
            result.pdf = analyticLightSelectionProbability / analyticLightAliasTable.weightSum;
        }
        else
        {
            result.pdf = 1.f;
        }

        return result;
    }

    EvaluatedLightSample evalLightSample(const LightSample lightSample, const float3 pos) { return loadLightSample(lightSample).eval(pos); }

    float3 evalEmission(const LightSample lightSample, float3 dir)
    {
        if (lightSample.isValid())
        {
            switch (lightSample.getKind())
            {
            case LightKind::Env:
                return evalEnvEmission(lightSample);
            case LightKind::Emissive:
                return evalEmissiveEmission(lightSample, dir);
            case LightKind::Analytic:
                return evalAnalyticEmission(lightSample);
            }
        }
        return float3(0.f);
    }

    float3 evalEnvEmission(const LightSample lightSample)
    {
        float3 dir = lightSample.getEnvWorldDir();
        return gScene.envMap.eval(dir);
    }

    float3 evalEmissiveEmission(const LightSample lightSample, float3 dir)
    {
        uint emissiveIndex = lightSample.getEmissiveIndex();
        const EmissiveTriangle tri = gScene.lightCollection.getTriangle(emissiveIndex);
        float3 emission;
        if (kUseEmissiveTextureForShading)
        {
            float2 texCoord = tri.getTexCoord(lightSample.getEmissiveBarycentrics());
            emission = gScene.materials.evalEmissive(tri.materialID, texCoord);
        }
        else
        {
            emission = gScene.lightCollection.getAverageRadiance(emissiveIndex);
        }

        if (gScene.materials.hasLightProfile(tri.materialID))
        {
            emission *= gScene.materials.evalLightProfile(dot(tri.normal, dir));
        }

        return emission;
    }

    float3 evalAnalyticEmission(const LightSample lightSample)
    {
        LightData light = gScene.getLight(lightSample.getAnalyticIndex());
        LightType type = LightType(light.type);
        switch (LightType(light.type))
        {
        case LightType::Point:
        case LightType::Directional:
        case LightType::Distant:
            return light.intensity;
        default:
            return float3(0.f);
        }
    }
};
