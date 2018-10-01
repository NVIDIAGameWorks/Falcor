/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "API/Texture.h"
#include "Data/Effects/CsmData.h"
#include "../Utils/GaussianBlur.h"
#include "Graphics/Light.h"
#include "Graphics/Scene/Scene.h"
#include "Utils/Math/ParallelReduction.h"
#include "Graphics/RenderGraph/RenderPass.h"

namespace Falcor
{
    class Gui;
    class CsmSceneRenderer;

    /** Cascaded Shadow Maps Technique
    */
    class CascadedShadowMaps : public RenderPass, public inherit_shared_from_this<RenderPass, CascadedShadowMaps>
    {
    public:
        using SharedPtr = std::shared_ptr<CascadedShadowMaps>;
        using UniquePtr = std::unique_ptr<CascadedShadowMaps>;

        enum class PartitionMode
        {
            Linear,
            Logarithmic,
            PSSM,
        };

        /** Destructor
        */
        ~CascadedShadowMaps();

        /** Create a new instance.
            \param[in] mapWidth Shadow map width
            \param[in] mapHeight Shadow map height
            \param[in] visibilityBufferWidth Visibility buffer width
            \param[in] visibilityBufferHeight Visibility buffer height
            \param[in] pLight Light to generate shadows for
            \param[in] pScene Scene to render when generating shadow maps
            \param[in] cascadeCount Number of cascades
            \param[in] shadowMapFormat Shadow map texture format
        */
        deprecate("3.2", "Use the other create() methods. Note that they now return a SharedPtr, and not UniquePtr")
        static UniquePtr create(uint32_t mapWidth, uint32_t mapHeight, uint32_t visibilityBufferWidth, uint32_t visibilityBufferHeight, Light::SharedConstPtr pLight, Scene::SharedPtr pScene, uint32_t cascadeCount = 4, uint32_t visMapBitsPerChannel = 16);
        static SharedPtr create(const Light::SharedConstPtr& pLight, uint32_t shadowMapWidth = 2048, uint32_t shadowMapHeight = 2048, uint32_t visibilityBufferWidth = 0, uint32_t visibilityBufferHeight = 0, const Scene::SharedPtr& pScene = nullptr, uint32_t cascadeCount = 4, uint32_t visMapBitsPerChannel = 16);
        static SharedPtr create(const Dictionary& dict = {});
        
        /** Render UI controls
            \param[in] pGui GUI instance to render UI elements with
            \param[in] uiGroup Optional name. If specified, UI elements will be rendered within a named group
        */
        void renderUI(Gui* pGui, const char* uiGroup = "") override;

        /** Run the shadow-map generation pass and the visibility pass. Returns the visibility buffer
            \params[in] pScene The scene to render
            \params[in] pCamera The camera that will be used to render the scene
            \params[in] pSceneDepthBuffer Valid only when SDSM is enabled. The depth map to run SDSM analysis on. If this is nullptr, SDSM will run a depth pass
        */
        Texture::SharedPtr generateVisibilityBuffer(RenderContext* pRenderCtx, const Camera* pCamera, const Texture::SharedPtr& pSceneDepthBuffer);

        /** Get the shadow map texture.
        */
        Texture::SharedPtr getShadowMap() const;

        /** Set number of cascade partitions.
        */
        void setCascadeCount(uint32_t cascadeCount);

        /** Get the number of partitions.
        */
        uint32_t getCascadeCount() { return mCsmData.cascadeCount; }

        /** Set whether to use SDSM.
        */
        void toggleMinMaxSdsm(bool enable) { mControls.useMinMaxSdsm = enable; }

        /** Set the min and max distance from the camera to generate shadows for.
        */
        void setDistanceRange(const glm::vec2& range) { mControls.distanceRange = range; }

        /** Set the filter mode. Options are defined in CsmData.h
        */
        void setFilterMode(uint32_t filterMode);

        /** Get the filter mode.
        */
        uint32_t getFilterMode() const { return mCsmData.filterMode; }

        /** Set the kernel width for PCF
        */
        void setPcfKernelWidth(uint32_t width) { mCsmData.pcfKernelWidth = width | 1; }

        /** Set the anistropy level for VSM/EVSM
        */
        void setVsmMaxAnisotropy(uint32_t maxAniso) { createVsmSampleState(maxAniso); }

        /** Set light-bleed reduction for VSM/EVSM
        */
        void setVsmLightBleedReduction(float reduction) { mCsmData.lightBleedingReduction = reduction; }

        /** Set the depth bias
        */
        void setDepthBias(float depthBias) { mCsmData.depthBias = depthBias; }

        /** Set the readback latency for SDSM (in frames)
        */
        void setSdsmReadbackLatency(uint32_t latency);

        /** Set the width and sigma used when blurring the EVSM shadow-map
        */
        void setEvsmBlur(uint32_t kernelWidth, float sigma);

        /** Enable mesh-culling for the shadow-map generation
        */
        void toggleMeshCulling(bool enabled);

        /** Check if mesh-culling is enabled
        */
        bool isMeshCullingEnabled() const;

        /** Enable saving cascade info into the gba channels of the visibility buffer
        */
        void toggleCascadeVisualization(bool shouldVisualze);

        /** Resize the visibility buffer
        */
        deprecate("3.2", "Use `onResize()` instead")
        void resizeVisibilityBuffer(uint32_t width, uint32_t height);

        /** Set the visibility's buffer bits-per-channel
        */
        void setVisibilityBufferBitsPerChannel(uint32_t bitsPerChannel);

        /** Reflect the render-pass
        */
        virtual RenderPassReflection reflect() const override;

        /** Execute the render-pass
        */
        virtual void execute(RenderContext* pContext, const RenderData* pRenderData) override;

        /** Set the scene
        */
        void setScene(const Scene::SharedPtr& pScene) override;

        /** Set the dimensions of the internal shadow-map
        */
        void resizeShadowMap(uint32_t width, uint32_t height);

        /** Set the light-source
        */
        void setLight(const Light::SharedConstPtr& pLight);

        /** Resize callback
        */
        virtual void onResize(uint32_t width, uint32_t height) override;
    private:
        CascadedShadowMaps(uint32_t mapWidth = 2048, uint32_t mapHeight = 2048);
        Light::SharedConstPtr mpLight;
        Camera::SharedPtr mpLightCamera;
        std::shared_ptr<CsmSceneRenderer> mpCsmSceneRenderer;
        std::shared_ptr<SceneRenderer> mpSceneRenderer;

        // Set shadow map generation parameters into a program.
        void setDataIntoGraphicsVars(GraphicsVars::SharedPtr pVars, const std::string& varName);
        vec2 calcDistanceRange(RenderContext* pRenderCtx, const Camera* pCamera, const Texture::SharedPtr& pDepthBuffer);
        void createDepthPassResources();
        void createShadowPassResources(uint32_t mapWidth, uint32_t mapHeight);
        void createVisibilityPassResources();
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

        GaussianBlur::UniquePtr mpGaussianBlur;

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
            FullScreenPass::UniquePtr pPass;
            GraphicsState::SharedPtr pState;
            GraphicsVars::SharedPtr pGraphicsVars;
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

        struct Controls
        {
            bool depthClamp = true;
            bool useMinMaxSdsm = true;
            glm::vec2 distanceRange = glm::vec2(0, 1);
            float pssmLambda = 0.5f;
            PartitionMode partitionMode = PartitionMode::Logarithmic;
            bool stabilizeCascades = false;
        };

        int32_t renderCascade = 0;
        Controls mControls;
        CsmData mCsmData;

        void setupVisibilityPassFbo(const Texture::SharedPtr& pVisBuffer);
        ProgramReflection::BindLocation mPerLightCbLoc;
        void executeInternal(RenderContext* pRenderCtx, const Camera* pCamera, const Texture::SharedPtr& pSceneDepthBuffer);
    };
}
