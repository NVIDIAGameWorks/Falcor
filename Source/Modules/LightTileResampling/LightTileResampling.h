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
#pragma once
#include "Utils/Sampling/AliasTable.h"
#include "Scene/Scene.h"
#include "Scene/Lights/ILightCollection.h"
#include "Scene/Lights/Light.h"
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

namespace Falcor
{

class LightTileResampling
{
public:
    struct Options
    {
        float envLightWeight = 1.f;
        float emissiveLightWeight = 1.f;
        float analyticLightWeight = 1.f;

        bool useEmissiveTextureForSampling = false;
        bool useEmissiveTextureForShading = true;

        uint32_t lightTileCount = 128;
        uint32_t lightTileSize = 1024;

        uint32_t screenTileSize = 8;
        uint32_t initialLightSampleCount = 32;

        Options() {}

        template<typename Archive>
        void serialize(Archive& ar)
        {
            ar("envLightWeight", envLightWeight);
            ar("emissiveLightWeight", emissiveLightWeight);
            ar("analyticLightWeight", analyticLightWeight);
            ar("useEmissiveTextureForSampling", useEmissiveTextureForSampling);
            ar("useEmissiveTextureForShading", useEmissiveTextureForShading);
            ar("lightTileCount", lightTileCount);
            ar("lightTileSize", lightTileSize);
            ar("screenTileSize", screenTileSize);
            ar("initialLightSampleCount", initialLightSampleCount);
        }
    };

    LightTileResampling(const ref<IScene>& pScene, const Options& options = Options());

    void beginFrame(RenderContext* pRenderContext);

    void generateLightTiles(RenderContext* pRenderContext);

    DefineList getDefines() const;

    void setLightsShaderData(const ShaderVar& var) const;

    ref<Buffer> getLightTileData() const { return mpLightTileData; }

    const Options& getOptions() const { return mOptions; }

    void setOptions(const Options& options);

    bool renderUI(Gui::Widgets& widget);

private:
    void prepareLighting(RenderContext* pRenderContext);
    void prepareResources();
    void updatePrograms();

    std::vector<float> computeEnvLightLuminance(RenderContext* pRenderContext, const ref<Texture>& texture);
    std::unique_ptr<AliasTable> buildEnvLightAliasTable(uint32_t width, uint32_t height, const std::vector<float>& luminances, std::mt19937& rng);
    std::unique_ptr<AliasTable> buildEmissiveLightAliasTable(RenderContext* pRenderContext, const ref<ILightCollection>& lightCollection, std::mt19937& rng);
    std::unique_ptr<AliasTable> buildAnalyticLightAliasTable(RenderContext* pRenderContext, const std::vector<ref<Light>>& lights, std::mt19937& rng);

    ref<IScene> mpScene;
    ref<Device> mpDevice;
    Options mOptions;

    sigs::Connection mUpdateFlagsConnection;
    IScene::UpdateFlags mUpdateFlags = IScene::UpdateFlags::None;

    std::mt19937 mRng;

    uint32_t mFrameIndex = 0;

    ref<ComputePass> mpReflectTypes;
    ref<ComputePass> mpGenerateLightTiles;

    ref<Buffer> mpEnvLightLuminance;
    float mEnvLightLuminanceFactor = 0.f;

    std::unique_ptr<AliasTable> mpEnvLightAliasTable;
    std::unique_ptr<AliasTable> mpEmissiveLightAliasTable;
    std::unique_ptr<AliasTable> mpAnalyticLightAliasTable;

    ref<Buffer> mpLightTileData;

    bool mRecompile = true;

    struct
    {
        float envLight = 0.f;
        float emissiveLights = 0.f;
        float analyticLights = 0.f;

        std::tuple<uint32_t, uint32_t, uint32_t> getSampleCount(uint32_t totalCount)
        {
            uint32_t envCount = (uint32_t)std::floor(envLight * totalCount);
            uint32_t emissiveCount = (uint32_t)std::floor(emissiveLights * totalCount);
            uint32_t analyticCount = (uint32_t)std::floor(analyticLights * totalCount);
            if (envCount > 0)
                envCount = totalCount - emissiveCount - analyticCount;
            else if (emissiveCount > 0)
                emissiveCount = totalCount - envCount - analyticCount;
            else if (analyticCount > 0)
                analyticCount = totalCount - envCount - emissiveCount;
            return {envCount, emissiveCount, analyticCount};
        }
    } mLightSelectionProbabilities;
};

} // namespace Falcor
