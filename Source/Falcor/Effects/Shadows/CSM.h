/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "RenderGraph/RenderGraph.h"
#include "RenderGraph/RenderPass.h"
#include "Data/Effects/CsmData.h"
#include "Core/Program/ProgramVars.h"
#include "Utils/Algorithm/ParallelReduction.h"

namespace Falcor
{
    class CsmSceneRenderer;

    /** Cascaded Shadow Maps Technique
    */
    class dlldecl CascadedShadowMaps : public RenderPass
    {
    public:
        using SharedPtr = std::shared_ptr<CascadedShadowMaps>;
        static const char* kDesc;

        enum class PartitionMode
        {
            Linear,
            Logarithmic,
            PSSM,
        };

        static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict = {});

        virtual std::string getDesc() override { return kDesc; }
        virtual Dictionary getScriptingDictionary() override;
        virtual RenderPassReflection reflect(const CompileData& compileData) override;
        virtual void compile(RenderContext* pContext, const CompileData& compileData) override;
        virtual void execute(RenderContext* pContext, const RenderData& renderData) override;
        virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
        virtual void renderUI(Gui::Widgets& widget) override;

        void toggleMeshCulling(bool enabled);

        // Scripting functions
        void setCascadeCount(uint32_t cascadeCount);
        void setMapWidth(uint32_t width);
        void setMapHeight(uint32_t height);
        void setVisibilityBufferBitsPerChannel(uint32_t bitsPerChannel);
        void setFilterMode(uint32_t filterMode);
        void setSdsmReadbackLatency(uint32_t latency);
        void setPartitionMode(uint32_t partitionMode) { mControls.partitionMode = static_cast<PartitionMode>(partitionMode); }
        void setPSSMLambda(float lambda) { mControls.pssmLambda = clamp(lambda, 0.f, 1.0f); }
        void setMinDistanceRange(float min) { mControls.distanceRange.x = clamp(min, 0.f, 1.f); }
        void setMaxDistanceRange(float max) { mControls.distanceRange.y = clamp(max, 0.f, 1.f); }
        void setCascadeBlendThreshold(float threshold) { mCsmData.cascadeBlendThreshold = clamp(threshold, 0.f, 1.0f); }
        void setDepthBias(float bias) { mCsmData.depthBias = max(0.f, bias); }
        void setPcfKernelWidth(uint32_t width) { mCsmData.pcfKernelWidth = width | 1; }
        void setVsmMaxAnisotropy(uint32_t maxAniso) { createVsmSampleState(maxAniso); }
        void setVsmLightBleedReduction(float reduction) { mCsmData.lightBleedingReduction = clamp(reduction, 0.f, 1.0f); }
        void setEvsmPositiveExponent(float exp) { mCsmData.evsmExponents.x = clamp(exp, 0.f, 5.54f); }
        void setEvsmNegativeExponent(float exp) { mCsmData.evsmExponents.y = clamp(exp, 0.f, 5.54f); }
        uint32_t getCascadeCount() { return mCsmData.cascadeCount; }
        uint32_t getMapWidth() { return mMapWidth; }
        uint32_t getMapHeight() { return mMapHeight; }
        uint32_t getVisibilityBufferBitsPerChannel() { return mVisibilityPassData.mapBitsPerChannel; }
        uint32_t getFilterMode() { return (uint32_t)mCsmData.filterMode; }
        uint32_t getSdsmReadbackLatency() { return mSdsmData.readbackLatency; }
        uint32_t getPartitionMode() { return (uint32_t)mControls.partitionMode; }
        float getPSSMLambda() { return mControls.pssmLambda; }
        float getMinDistanceRange() { return mControls.distanceRange.x; }
        float getMaxDistanceRange() { return mControls.distanceRange.y; }
        float getCascadeBlendThreshold() { return mCsmData.cascadeBlendThreshold; }
        float getDepthBias() { return mCsmData.depthBias; }
        uint32_t getPcfKernelWidth() { return mCsmData.pcfKernelWidth; }
        uint32_t getVsmMaxAnisotropy() { return mShadowPass.pVSMTrilinearSampler->getMaxAnisotropy(); }
        float getVsmLightBleedReduction() { return mCsmData.lightBleedingReduction; }
        float getEvsmPositiveExponent() { return mCsmData.evsmExponents.x; }
        float getEvsmNegativeExponent() { return mCsmData.evsmExponents.y; }

    private:
        CascadedShadowMaps();
        uint32_t mMapWidth = 2048;
        uint32_t mMapHeight = 2048;
        Light::SharedConstPtr mpLight;
        Camera::SharedPtr mpLightCamera;
        //std::shared_ptr<CsmSceneRenderer> mpCsmSceneRenderer;
        Scene::SharedPtr mpScene;

        void createDepthPassResources();
        void createShadowPassResources();
        void createVisibilityPassResources();

        // Set shadow map generation parameters into a program.
        void setDataIntoVars(const GraphicsVars::SharedPtr& pVars, const std::string& varName);
        vec2 calcDistanceRange(RenderContext* pRenderCtx, const Camera* pCamera, const Texture::SharedPtr& pDepthBuffer);
        void partitionCascades(const Camera* pCamera, const glm::vec2& distanceRange);
        void renderScene(RenderContext* pCtx);

        // Shadow-pass
        struct
        {
            Fbo::SharedPtr pFbo;
            float fboAspectRatio;
            Sampler::SharedPtr pPointCmpSampler;
            Sampler::SharedPtr pLinearCmpSampler;
            Sampler::SharedPtr pVSMTrilinearSampler;
            GraphicsVars::SharedPtr pGraphicsVars;
            GraphicsState::SharedPtr pState;
            glm::vec2 mapSize;
        } mShadowPass;

        // SDSM
        struct SdsmData
        {
            ParallelReduction::UniquePtr minMaxReduction;
            vec2 sdsmResult;   // Used for displaying the range in the UI
            uint32_t width = 0;
            uint32_t height = 0;
            uint32_t sampleCount = 0;
            int32_t readbackLatency = 1;
        };
        SdsmData mSdsmData;
        void createSdsmData(Texture::SharedPtr pTexture);
        void reduceDepthSdsmMinMax(RenderContext* pRenderCtx, const Camera* pCamera, const Texture::SharedPtr pDepthBuffer);
        void createVsmSampleState(uint32_t maxAnisotropy);

        RenderGraph::SharedPtr mpBlurGraph;
        Dictionary mBlurDict;

        // Depth-pass
        struct
        {
            GraphicsState::SharedPtr pState;
            GraphicsVars::SharedPtr pGraphicsVars;
        } mDepthPass;
        void executeDepthPass(RenderContext* pCtx, const Camera* pCamera);

        //Visibility pass
        struct
        {
            FullScreenPass::SharedPtr pPass;
            Fbo::SharedPtr pFbo;
            uint32_t mVisualizeCascadesOffset;
        } mVisibilityPass;

        struct
        {
            //This is effectively a bool, but bool only takes up 1 byte which messes up setBlob
            uint32_t shouldVisualizeCascades = 0u;
            int3 padding;
            glm::mat4 camInvViewProj;
            glm::uvec2 screenDim = { 0, 0 };
            uint32_t mapBitsPerChannel = 32;
        } mVisibilityPassData;

        struct
        {
            bool depthClamp = true;
            bool useMinMaxSdsm = true;
            glm::vec2 distanceRange = glm::vec2(0, 1);
            float pssmLambda = 0.5f;
            PartitionMode partitionMode = PartitionMode::Logarithmic;
            bool stabilizeCascades = false;
        } mControls;

        int32_t renderCascade = 0;
        CsmData mCsmData;
        bool mCullMeshes = true;

        /** Resize
        */
        void onResize(uint32_t width, uint32_t height);
        void setupVisibilityPassFbo(const Texture::SharedPtr& pVisBuffer);
        ProgramReflection::BindLocation mPerLightCbLoc;

        void resizeShadowMap(uint32_t width, uint32_t height);
        void setLight(const Light::SharedConstPtr& pLight);
    };
}
