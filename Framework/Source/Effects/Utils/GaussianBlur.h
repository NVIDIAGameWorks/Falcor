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
#include "API/FBO.h"
#include "Graphics/FullScreenPass.h"
#include "API/Sampler.h"
#include "Graphics/Program/ProgramVars.h"
#include <memory>

namespace Falcor
{
    class RenderContext;
    class Texture;
    class Fbo;
    class Gui;

    /** Gaussian-blur technique
    */
    class GaussianBlur
    {
    public:
        using UniquePtr = std::unique_ptr<GaussianBlur>;
        /** Destructor
        */
        ~GaussianBlur();

        /** Create a new object
            \param[in] kernelSize Number of samples taken along each axis
            \param[in] sigma Gaussian distribution sigma value used to calculate sample weights. Values smaller than twice the sigma are ineffective
        */
        static UniquePtr create(uint32_t kernelSize = 5, float sigma = 1.0f);

        /** Apply gaussian blur by rendering one texture into another.
            \param pRenderContext Render context to use
            \param pSrc The source texture
            \param pDst The destination texture
        */
        void execute(RenderContext* pRenderContext, Texture::SharedPtr pSrc, Fbo::SharedPtr pDst);

        /** Set the kernel width. Controls the number of texels which will be sampled when blurring each pixel.
            Values smaller than twice the sigma are ineffective.
        */
        void setKernelWidth(uint32_t kernelWidth);

        /** Get the kernel width
        */
        uint32_t getKernelWidth() const { return mKernelWidth; }

        /** Set the sigma. Higher values will result in a blurrier image.
            Values greater than half the kernel width are ineffective.
        */
        void setSigma(float sigma) { mSigma = sigma; mDirty = true; }

        /** Get the sigma value.
        */
        float getSigma() const { return mSigma; }

        /** Render UI controls for blur settings.
            \param[in] pGui GUI instance to render UI elements with
            \param[in] uiGroup Optional name. If specified, UI elements will be rendered within a named group
        */
        void renderUI(Gui* pGui, const char* uiGroup = nullptr);

    private:
        GaussianBlur(uint32_t kernelSize, float sigma);
        uint32_t mKernelWidth;
        float mSigma;
        uint32_t vpMask;
        void createTmpFbo(const Texture* pSrc);
        void createProgram();
        void updateKernel();

        FullScreenPass::UniquePtr mpHorizontalBlur;
        FullScreenPass::UniquePtr mpVerticalBlur;
        Fbo::SharedPtr mpTmpFbo;
        Sampler::SharedPtr mpSampler;
        bool mDirty = true;
        GraphicsVars::SharedPtr mpVars;

        struct
        {
            ParameterBlockReflection::BindLocation sampler;
            ParameterBlockReflection::BindLocation srcTexture;
        } mBindLocations;
    };
}