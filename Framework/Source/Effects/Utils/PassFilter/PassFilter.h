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
#include "API/FBO.h"
#include "Graphics/FullScreenPass.h"
#include "API/Sampler.h"
#include "Graphics/Program/ProgramVars.h"
#include <memory>

namespace Falcor
{
    class RenderContext;

    class PassFilter
    {
    public:
        using SharedPtr = std::shared_ptr<PassFilter>;

        enum class Type
        {
            HighPass,
            LowPass
        };

        /** Create a new object
            \param[in] threshold Brightness threshold to pass through the filter
        */
        static SharedPtr create(Type filterType, float threshold);

        /** Apply the filter to a texture
            \param pRenderContext Render context to use
            \param pSrc The source texture
            \return A texture containing filter results
        */
        Texture::SharedPtr execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrc);

        /** Apply the filter to a texture
            \param pRenderContext Render context to use
            \param pSrc The source texture
            \param pDst Fbo to write results to
            \return A texture containing filter results
        */
        Texture::SharedPtr execute(RenderContext* pRenderContext, const Texture::SharedPtr& pSrc, const Fbo::SharedPtr& pDst);

        /** Set the threshold value for the filter.
        */
        void setThreshold(float threshold) { mThreshold = threshold; mDirty = true; }

        /** Get the filter threshold value
        */
        float getThreshold() const { return mThreshold; }

    private:
        PassFilter(Type filterType, float threshold);
        void updateResultFbo(const Texture* pSrc);
        void initProgram();

        Type mFilterType;
        float mThreshold;
        bool mDirty = true;

        Fbo::SharedPtr mpResultFbo;
        Sampler::SharedPtr mpSampler;

        FullScreenPass::UniquePtr mpFilterPass;
        GraphicsVars::SharedPtr mpVars;
        ConstantBuffer::SharedPtr mpParamCB;

        struct
        {
            ParameterBlockReflection::BindLocation sampler;
            ParameterBlockReflection::BindLocation srcTexture;
        } mBindLocations;
    };
}