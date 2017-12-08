/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "API/FBO.h"
#include "Graphics/Program/ProgramVars.h"
#include "Graphics/FullScreenPass.h"
#include "Data/Effects/SSAOData.h"
#include "Effects/Utils/GaussianBlur.h"
#include "Utils/Gui.h"

namespace Falcor
{
    class Gui;
    class Camera;

    class SSAO
    {
    public:
        using UniquePtr = std::unique_ptr<SSAO>;

        enum class SampleDistribution
        {
            Random,
            UniformHammersley,
            CosineHammersley
        };

        /** Create an SSAO pass object sampling with a hemisphere kernel by default.
            \param[in] aoMapSize Width and height of the AO map texture
            \param[in] kernelSize Number of samples in the AO kernel
            \param[in] blurSize Kernel size used for blurring the AO map
            \param[in] blurSigma Sigma of the blur kernel
            \param[in] noiseSize Width and height of the noise texture
            \param[in] distribution Distribution of sample points when using a hemisphere kernel.
            \return SSAO pass object.
        */
        static UniquePtr create(const uvec2& aoMapSize, uint32_t kernelSize = 16, uint32_t blurSize = 5, float blurSigma = 2.0f, const uvec2& noiseSize = uvec2(16), SampleDistribution distribution = SampleDistribution::CosineHammersley);

        /** Render GUI for tweaking SSAO settings
        */
        void renderGui(Gui* pGui);

        /** Generate the AO map
            \param[in] pContext Render context
            \param[in] pCamera Camera used to render the scene
            \param[in] pDepthTexture Scene depth buffer
            \param[in] pNormalTexture Scene world-space normals buffer
            \return AO map texture
        */
        Texture::SharedPtr generateAOMap(RenderContext* pContext, const Camera* pCamera, const Texture::SharedPtr& pDepthTexture, const Texture::SharedPtr& pNormalTexturer);

        /** Sets blur kernel size
        */
        void setBlurKernelWidth(uint32_t width) { mpBlur->setKernelWidth(width); }

        /** Set blur sigma value
        */
        void setBlurSigma(float sigma) { mpBlur->setSigma(sigma); }

        /** Recreate sampling kernel
            \param[in] kernelSize Number of samples
            \param[in] distribution Distribution of sample points within a hemisphere kernel. Parameter is ignored for sphere kernel generation, but is saved for use in future hemisphere kernels.
        */
        void setKernel(uint32_t kernelSize, SampleDistribution distribution = SampleDistribution::Random);

        /** Recreate noise texture
            \param[in] width Noise texture width
            \param[in] height Noise texture height
        */
        void setNoiseTexture(uint32_t width, uint32_t height);

    private:

        SSAO(const uvec2& aoMapSize, uint32_t kernelSize, uint32_t blurSize, float blurSigma, const uvec2& noiseSize, SampleDistribution distribution);

        void upload();

        void initShader();

        SSAOData mData;
        bool mDirty = false;

        Fbo::SharedPtr mpAOFbo;
        GraphicsState::SharedPtr mpSSAOState;
        Sampler::SharedPtr mpNoiseSampler;
        Texture::SharedPtr mpNoiseTexture;

        Sampler::SharedPtr mpTextureSampler;

        struct
        {
            ProgramReflection::BindLocation internalPerFrameCB;
            ProgramReflection::BindLocation ssaoCB;
            ProgramReflection::BindLocation noiseSampler;
            ProgramReflection::BindLocation textureSampler;
            ProgramReflection::BindLocation depthTex;
            ProgramReflection::BindLocation normalTex;
            ProgramReflection::BindLocation noiseTex;
        } mBindLocations;

        uint32_t mHemisphereDistribution = (uint32_t)SampleDistribution::CosineHammersley;

        static const Gui::DropdownList kKernelDropdown;
        static const Gui::DropdownList kDistributionDropdown;

        FullScreenPass::UniquePtr mpSSAOPass;
        GraphicsVars::SharedPtr mpSSAOVars;

        bool mApplyBlur = true;
        GaussianBlur::UniquePtr mpBlur;
    };

}