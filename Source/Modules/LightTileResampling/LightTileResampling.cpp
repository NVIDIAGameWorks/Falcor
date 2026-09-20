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
#include "LightTileResampling.h"
#include "Core/Error.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Timing/Profiler.h"
#include "Utils/Color/ColorHelpers.slang"

namespace Falcor
{

namespace
{
const char kReflectTypesFile[] = "LightTileResampling/ReflectTypes.cs.slang";
const char kGenerateLightTilesFile[] = "LightTileResampling/GenerateLightTiles.cs.slang";
} // namespace

LightTileResampling::LightTileResampling(const ref<IScene>& pScene, const Options& options)
    : mpScene(pScene), mpDevice(pScene->getDevice()), mOptions(options)
{
    FALCOR_ASSERT(mpScene);

    ProgramDesc desc;
    DefineList defines;
    mpScene->getShaderDefines(defines);
    defines.add(getDefines());
    desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
    mpReflectTypes = ComputePass::create(mpDevice, desc, defines);

    mUpdateFlagsConnection = mpScene->getUpdateFlagsSignal().connect([&](IScene::UpdateFlags flags) { mUpdateFlags |= flags; });
}

void LightTileResampling::setOptions(const Options& options)
{
    if (std::memcmp(&mOptions, &options, sizeof(Options)) != 0)
    {
        mOptions = options;
        mRecompile = true;
    }
}

bool LightTileResampling::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto group = widget.group("Light Tile Resampling"))
    {
        if (auto lwGroup = group.group("Light selection weights"))
        {
            mRecompile |= lwGroup.var("Environment", mOptions.envLightWeight, 0.f, 1.f);
            mRecompile |= lwGroup.var("Emissive", mOptions.emissiveLightWeight, 0.f, 1.f);
            mRecompile |= lwGroup.var("Analytic", mOptions.analyticLightWeight, 0.f, 1.f);
        }

        mRecompile |= group.checkbox("Emissive texture for sampling", mOptions.useEmissiveTextureForSampling);
        mRecompile |= group.checkbox("Emissive texture for shading", mOptions.useEmissiveTextureForShading);

        mRecompile |= group.var("Light tile count", mOptions.lightTileCount, 1u, 1024u);
        mRecompile |= group.var("Light tile size", mOptions.lightTileSize, 1u, 8192u);
        mRecompile |= group.var("Screen tile size", mOptions.screenTileSize, 1u, 32u);
        mRecompile |= group.var("Initial light sample count", mOptions.initialLightSampleCount, 1u, 1024u);
    }

    dirty |= mRecompile;

    return dirty;
}

void LightTileResampling::beginFrame(RenderContext* pRenderContext)
{
    if (is_set(mUpdateFlags, IScene::UpdateFlags::RecompileNeeded) || is_set(mUpdateFlags, IScene::UpdateFlags::GeometryChanged))
        mRecompile = true;

    prepareLighting(pRenderContext);
    prepareResources();

    if (mRecompile)
        updatePrograms();

    mUpdateFlags = IScene::UpdateFlags::None;
}

void LightTileResampling::generateLightTiles(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "LightTileResampling::generateLightTiles");

    auto rootVar = mpGenerateLightTiles->getRootVar();
    mpScene->bindShaderData(rootVar["gScene"]);

    auto var = rootVar["CB"]["gGenerateLightTiles"];
    var["lightTileData"] = mpLightTileData;
    setLightsShaderData(var["lights"]);
    var["frameIndex"] = mFrameIndex;

    mpGenerateLightTiles->execute(pRenderContext, uint3(mOptions.lightTileSize, mOptions.lightTileCount, 1));
    mFrameIndex++;
}

DefineList LightTileResampling::getDefines() const
{
    DefineList defines;

    // Tile and initial sampling defines.
    defines.add("LIGHT_TILE_COUNT", std::to_string(mOptions.lightTileCount));
    defines.add("LIGHT_TILE_SIZE", std::to_string(mOptions.lightTileSize));
    defines.add("SCREEN_TILE_SIZE", std::to_string(mOptions.screenTileSize));
    defines.add("INITIAL_LIGHT_SAMPLE_COUNT", std::to_string(mOptions.initialLightSampleCount));

    // Light sample bit allocation defines.
    uint32_t envIndexBits = 26, envPositionBits = 4;
    uint32_t emissiveIndexBits = 22, emissivePositionBits = 8;
    uint32_t analyticIndexBits = 14, analyticPositionBits = 16;

    auto computeIndexPositionBits = [](AliasTable& aliasTable, uint32_t& indexBits, uint32_t& positionBits)
    {
        uint32_t count = aliasTable.getCount();
        indexBits = 0;
        while (count > 0)
        {
            ++indexBits;
            count >>= 1;
        }
        if (indexBits & 1)
            ++indexBits;
        if (indexBits >= 30)
            FALCOR_THROW("Count too large to be represented in 30 bits");
        positionBits = 30 - indexBits;
    };

    if (mpEnvLightAliasTable)
        computeIndexPositionBits(*mpEnvLightAliasTable, envIndexBits, envPositionBits);
    if (mpEmissiveLightAliasTable)
        computeIndexPositionBits(*mpEmissiveLightAliasTable, emissiveIndexBits, emissivePositionBits);
    if (mpAnalyticLightAliasTable)
        computeIndexPositionBits(*mpAnalyticLightAliasTable, analyticIndexBits, analyticPositionBits);

    defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");

    defines.add("LIGHT_SAMPLE_ENV_INDEX_BITS", std::to_string(envIndexBits));
    defines.add("LIGHT_SAMPLE_ENV_POSITION_BITS", std::to_string(envPositionBits));
    defines.add("LIGHT_SAMPLE_EMISSIVE_INDEX_BITS", std::to_string(emissiveIndexBits));
    defines.add("LIGHT_SAMPLE_EMISSIVE_POSITION_BITS", std::to_string(emissivePositionBits));
    defines.add("LIGHT_SAMPLE_ANALYTIC_INDEX_BITS", std::to_string(analyticIndexBits));
    defines.add("LIGHT_SAMPLE_ANALYTIC_POSITION_BITS", std::to_string(analyticPositionBits));

    defines.add("USE_EMISSIVE_TEXTURE_FOR_SAMPLING", mOptions.useEmissiveTextureForSampling ? "1" : "0");
    defines.add("USE_EMISSIVE_TEXTURE_FOR_SHADING", mOptions.useEmissiveTextureForShading ? "1" : "0");
    defines.add("USE_LOCAL_EMISSIVE_TRIANGLES", "0");

    return defines;
}

void LightTileResampling::setLightsShaderData(const ShaderVar& var) const
{
    var["envLightLuminance"] = mpEnvLightLuminance;

    if (mpEnvLightAliasTable)
        mpEnvLightAliasTable->bindShaderData(var["envLightAliasTable"]);
    if (mpEmissiveLightAliasTable)
        mpEmissiveLightAliasTable->bindShaderData(var["emissiveLightAliasTable"]);
    if (mpAnalyticLightAliasTable)
        mpAnalyticLightAliasTable->bindShaderData(var["analyticLightAliasTable"]);

    var["envLightLuminanceFactor"] = mEnvLightLuminanceFactor;

    var["envLightSelectionProbability"] = mLightSelectionProbabilities.envLight;
    var["emissiveLightSelectionProbability"] = mLightSelectionProbabilities.emissiveLights;
    var["analyticLightSelectionProbability"] = mLightSelectionProbabilities.analyticLights;
}

void LightTileResampling::prepareLighting(RenderContext* pRenderContext)
{
    if (is_set(mUpdateFlags, IScene::UpdateFlags::RenderSettingsChanged))
        mRecompile = true;

    if (mpScene->useEnvLight())
    {
        if (is_set(mUpdateFlags, IScene::UpdateFlags::EnvMapChanged))
            mpEnvLightAliasTable = nullptr;

        const auto& envMap = mpScene->getEnvMap();
        if (!mpEnvLightLuminance || !mpEnvLightAliasTable)
        {
            const auto& texture = envMap->getEnvMap();
            auto luminances = computeEnvLightLuminance(pRenderContext, texture);
            mpEnvLightLuminance = mpDevice->createTypedBuffer<float>(
                (uint32_t)luminances.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, luminances.data()
            );
            mpEnvLightAliasTable = buildEnvLightAliasTable(texture->getWidth(), texture->getHeight(), luminances, mRng);
            mRecompile = true;
        }

        mEnvLightLuminanceFactor = luminance(envMap->getIntensity() * envMap->getTint());
    }
    else
    {
        if (mpEnvLightLuminance)
        {
            mpEnvLightLuminance = nullptr;
            mpEnvLightAliasTable = nullptr;
            mRecompile = true;
        }
    }

    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        if (is_set(mUpdateFlags, IScene::UpdateFlags::GeometryChanged))
            mpEmissiveLightAliasTable = nullptr;
        if (!mpEmissiveLightAliasTable)
        {
            auto lightCollection = mpScene->getILightCollection(pRenderContext);
            lightCollection->update(pRenderContext);
            if (lightCollection->getActiveLightCount(pRenderContext) > 0)
            {
                mpEmissiveLightAliasTable = buildEmissiveLightAliasTable(pRenderContext, lightCollection, mRng);
                mRecompile = true;
            }
        }
    }
    else
    {
        if (mpEmissiveLightAliasTable)
        {
            mpEmissiveLightAliasTable = nullptr;
            mRecompile = true;
        }
    }

    if (mpScene->useAnalyticLights())
    {
        if (is_set(mUpdateFlags, IScene::UpdateFlags::LightCountChanged))
            mpAnalyticLightAliasTable = nullptr;
        if (!mpAnalyticLightAliasTable)
        {
            std::vector<ref<Light>> lights = mpScene->getActiveAnalyticLights();
            if (!lights.empty())
            {
                mpAnalyticLightAliasTable = buildAnalyticLightAliasTable(pRenderContext, lights, mRng);
                mRecompile = true;
            }
        }
    }
    else
    {
        if (mpAnalyticLightAliasTable)
        {
            mpAnalyticLightAliasTable = nullptr;
            mRecompile = true;
        }
    }

    auto& probs = mLightSelectionProbabilities;
    probs.envLight = mpEnvLightAliasTable ? mOptions.envLightWeight : 0.f;
    probs.emissiveLights = mpEmissiveLightAliasTable ? mOptions.emissiveLightWeight : 0.f;
    probs.analyticLights = mpAnalyticLightAliasTable ? mOptions.analyticLightWeight : 0.f;
    float total = probs.envLight + probs.emissiveLights + probs.analyticLights;
    if (total > 0.f)
    {
        probs.envLight /= total;
        probs.emissiveLights /= total;
        probs.analyticLights /= total;
    }
}

void LightTileResampling::prepareResources()
{
    auto reflectVar = mpReflectTypes->getRootVar();

    uint32_t elementCount = mOptions.lightTileCount * mOptions.lightTileSize;
    if (!mpLightTileData || mpLightTileData->getElementCount() < elementCount)
    {
        mpLightTileData = mpDevice->createStructuredBuffer(
            reflectVar["lightTileData"],
            elementCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }
}

void LightTileResampling::updatePrograms()
{
    DefineList commonDefines;
    mpScene->getShaderDefines(commonDefines);
    commonDefines.add(getDefines());
    commonDefines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    commonDefines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    commonDefines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");

    // GenerateLightTiles-specific defines.
    DefineList defines = commonDefines;
    auto [envLightSampleCount, emissiveLightSampleCount, analyticLightSampleCount] =
        mLightSelectionProbabilities.getSampleCount(mOptions.lightTileSize);
    defines.add("ENV_LIGHT_SAMPLE_COUNT", std::to_string(envLightSampleCount));
    defines.add("EMISSIVE_LIGHT_SAMPLE_COUNT", std::to_string(emissiveLightSampleCount));
    defines.add("ANALYTIC_LIGHT_SAMPLE_COUNT", std::to_string(analyticLightSampleCount));

    if (!mpGenerateLightTiles)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kGenerateLightTilesFile).csEntry("main");
        mpGenerateLightTiles = ComputePass::create(mpDevice, desc, defines, false);
    }

    mpGenerateLightTiles->getProgram()->addDefines(defines);
    mpGenerateLightTiles->setVars(nullptr);

    mRecompile = false;
}

std::vector<float> LightTileResampling::computeEnvLightLuminance(RenderContext* pRenderContext, const ref<Texture>& texture)
{
    FALCOR_ASSERT(texture);

    uint32_t width = texture->getWidth();
    uint32_t height = texture->getHeight();

    std::vector<uint8_t> texelsRaw;
    if (getFormatType(texture->getFormat()) == FormatType::Float)
    {
        texelsRaw = pRenderContext->readTextureSubresource(texture.get(), 0);
    }
    else
    {
        auto floatTexture = mpDevice->createTexture2D(
            width, height, ResourceFormat::RGBA32Float, 1, 1, nullptr, ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
        );
        pRenderContext->blit(texture->getSRV(), floatTexture->getRTV());
        texelsRaw = pRenderContext->readTextureSubresource(floatTexture.get(), 0);
    }

    uint32_t texelCount = width * height;
    uint32_t channelCount = getFormatChannelCount(texture->getFormat());
    const float* texels = reinterpret_cast<const float*>(texelsRaw.data());

    std::vector<float> luminances(texelCount);

    if (channelCount == 1)
    {
        for (uint32_t i = 0; i < texelCount; ++i)
        {
            luminances[i] = texels[0];
            texels += channelCount;
        }
    }
    else if (channelCount == 3 || channelCount == 4)
    {
        for (uint32_t i = 0; i < texelCount; ++i)
        {
            luminances[i] = luminance(float3(texels[0], texels[1], texels[2]));
            texels += channelCount;
        }
    }
    else
    {
        FALCOR_THROW("Invalid number of channels in env map");
    }

    return luminances;
}

std::unique_ptr<AliasTable> LightTileResampling::buildEnvLightAliasTable(
    uint32_t width,
    uint32_t height,
    const std::vector<float>& luminances,
    std::mt19937& rng
)
{
    FALCOR_ASSERT(luminances.size() == width * height);

    std::vector<float> weights(width * height);

    for (uint32_t i = 0, y = 0; y < height; ++y)
    {
        float theta = (float)M_PI * (y + 0.5f) / height;
        float solidAngle = (2.f * (float)M_PI / width) * ((float)M_PI / height) * std::sin(theta);

        for (uint32_t x = 0; x < width; ++x, ++i)
        {
            weights[i] = luminances[i] * solidAngle;
        }
    }

    return std::make_unique<AliasTable>(mpDevice, std::move(weights), rng);
}

std::unique_ptr<AliasTable> LightTileResampling::buildEmissiveLightAliasTable(
    RenderContext* pRenderContext,
    const ref<ILightCollection>& lightCollection,
    std::mt19937& rng
)
{
    FALCOR_ASSERT(lightCollection);

    lightCollection->update(pRenderContext);

    const auto& triangles = lightCollection->getMeshLightTriangles(pRenderContext);

    std::vector<float> weights(triangles.size());

    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = luminance(triangles[i].averageRadiance) * triangles[i].area;
    }

    return std::make_unique<AliasTable>(mpDevice, std::move(weights), rng);
}

std::unique_ptr<AliasTable> LightTileResampling::buildAnalyticLightAliasTable(
    RenderContext* pRenderContext,
    const std::vector<ref<Light>>& lights,
    std::mt19937& rng
)
{
    std::vector<float> weights(lights.size());

    for (size_t i = 0; i < weights.size(); ++i)
    {
        weights[i] = 1.f;
    }

    return std::make_unique<AliasTable>(mpDevice, std::move(weights), rng);
}

} // namespace Falcor
