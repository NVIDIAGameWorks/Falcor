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

namespace Falcor
{
    class Gui;
    class CsmSceneRenderer;

    /** Cascaded Shadow Maps Technique
    */
    class CascadedShadowMaps
    {
    public:
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
            \param[in] pLight Light to generate shadows for
            \param[in] pScene Scene to render when generating shadow maps
            \param[in] cascadeCount Number of cascades
            \param[in] shadowMapFormat Shadow map texture format
        */
        static UniquePtr create(uint32_t mapWidth, uint32_t mapHeight, Light* pLight, Scene* pScene, uint32_t cascadeCount = 4, ResourceFormat shadowMapFormat = ResourceFormat::D32Float);

        /** Render UI controls
            \param[in] pGui GUI instance to render UI elements with
            \param[in] uiGroup Optional name. If specified, UI elements will be rendered within a named group
        */
        void renderUi(Gui* pGui, const char* uiGroup = nullptr);

        /** Run the shadow-map generation pass
            \params[in] pScene The scene to render
            \params[in] pCamera The camera that will be used to render the scene
            \params[in] pSceneDepthBuffer Valid only when SDSM is enabled. The depth map to run SDSM analysis on. If this is nullptr, SDSM will run a depth pass
        */
        void setup(RenderContext* pRenderCtx, const Camera* pCamera, Texture::SharedPtr pSceneDepthBuffer);

        /** Get the shadow map texture.
        */
        Texture::SharedPtr getShadowMap() const;

        /** Set shadow map generation parameters into a program.
            \param[in] pVars GraphicsVars of the program to set data into
            \param[in] varName Name of the CsmData variable in the program
        */
        void setDataIntoGraphicsVars(GraphicsVars::SharedPtr pVars, const std::string& varName);
        void setDataIntoParameterBlock(ParameterBlock* pBlock, ConstantBuffer::SharedPtr pCb, size_t offset, const std::string & varName);
        void setDataIntoParameterBlock(ParameterBlock* pBlock, const std::string& varName);

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

        void setPcfKernelWidth(uint32_t width) { mCsmData.pcfKernelWidth = width | 1; }

        void setVsmMaxAnisotropy(uint32_t maxAniso) { createVsmSampleState(maxAniso); }

        void setVsmLightBleedReduction(float reduction) { mCsmData.lightBleedingReduction = reduction; }

        void setDepthBias(float depthBias) { mCsmData.depthBias = depthBias; }

        void setSdsmReadbackLatency(uint32_t latency);
        void setEvsmBlur(uint32_t kernelWidth, float sigma);
    private:
        CascadedShadowMaps(uint32_t mapWidth, uint32_t mapHeight, Light* pLight, Scene* pScene, uint32_t cascadeCount, ResourceFormat shadowMapFormat);
        Light* mpLight;
        Scene* mpScene;
        Camera::SharedPtr mpLightCamera;
        std::shared_ptr<CsmSceneRenderer> mpCsmSceneRenderer;
        std::shared_ptr<SceneRenderer> mpSceneRenderer;

        vec2 calcDistanceRange(RenderContext* pRenderCtx, const Camera* pCamera, Texture::SharedPtr pDepthBuffer);
        void createShadowPassResources(uint32_t mapWidth, uint32_t mapHeight);
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
            int32_t readbackLatency = 1;
        };
        SdsmData mSdsmData;
        void createSdsmData(Texture::SharedPtr pTexture);
        void reduceDepthSdsmMinMax(RenderContext* pRenderCtx, const Camera* pCamera, Texture::SharedPtr pDepthBuffer);
        void createVsmSampleState(uint32_t maxAnisotropy);

        GaussianBlur::UniquePtr mpGaussianBlur;

        // Depth-pass
        struct
        {
            GraphicsState::SharedPtr pState;
            GraphicsVars::SharedPtr pGraphicsVars;
        } mDepthPass;
        void executeDepthPass(RenderContext* pCtx, const Camera* pCamera);

        struct Controls
        {
            bool depthClamp = true;
            bool useMinMaxSdsm = false;
            glm::vec2 distanceRange = glm::vec2(0, 1);
            float pssmLambda = 0.8f;
            PartitionMode partitionMode = PartitionMode::PSSM;
            bool stabilizeCascades = false;
        };

        int32_t renderCascade = 0;
        Controls mControls;
        CsmData mCsmData;

        ProgramReflection::BindLocation mPerLightCbLoc;
    };
}
