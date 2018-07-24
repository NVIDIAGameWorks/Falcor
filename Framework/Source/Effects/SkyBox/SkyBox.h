/***************************************************************************
# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
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
#include <memory>
#include "API/Sampler.h"
#include "API/Texture.h"
#include "Graphics/Model/Model.h"
#include "Graphics/Program/Program.h"
#include "API/ConstantBuffer.h"
#include "API/DepthStencilState.h"
#include "API/RasterizerState.h"
#include "API/BlendState.h"
#include "Graphics/RenderGraph/RenderPass.h"

namespace Falcor
{
    class RenderContext;

    class SkyBox : public RenderPass, inherit_shared_from_this<RenderPass, SkyBox>
    {
    public:
        using SharedPtr = std::shared_ptr<SkyBox>;
        using UniquePtr = std::unique_ptr<SkyBox>;

        /** Create a sky box using an existing texture
            \param[in] pTexture Sky box texture
            \param[in] pSampler Sampler to use when rendering this sky box
            \param[in] renderStereo Whether to render in stereo mode using Single Pass Stereo
        */
        static UniquePtr create(Texture::SharedPtr& pTexture, Sampler::SharedPtr pSampler = nullptr, bool renderStereo = false);
        static UniquePtr deserialize(const RenderPassSerializer& serializer) 
        {
            std::string skyBox = getDirectoryFromFile(serializer.getValue("gSceneFilename").str) + '/' + serializer.getValue("gSkyBoxFilename").str;
            Sampler::Desc samplerDesc; 
            samplerDesc.setFilterMode(static_cast<Falcor::Sampler::Filter>(serializer.getValue("sampleDesc.minFilter").i32), 
                static_cast<Falcor::Sampler::Filter>(serializer.getValue("sampleDesc.magFilter").i32), 
                static_cast<Falcor::Sampler::Filter>(serializer.getValue("sampleDesc.mipFilter").i32));
            return createFromTexture(skyBox, serializer.getValue("loadAsSrgb").b, Sampler::create(samplerDesc));
        }

        /** Load a texture and create a sky box using it.
            \param[in] textureName Filename of texture. Can include a full or relative path from a data directory
            \param[in] loadAsSrgb Whether to load the texture into an sRGB format
            \param[in] pSampler Sampler to use when rendering this sky box
            \param[in] renderStereo Whether to render in stereo mode using Single Pass Stereo
        */
        static UniquePtr createFromTexture(const std::string& textureName, bool loadAsSrgb = true, Sampler::SharedPtr pSampler = nullptr, bool renderStereo = false);

        /** Render the sky box.
            \param[in] pRenderCtx Render context
            \param[in] pCamera Camera to use when rendering
            \param[in, optional] The target FBO. If this is nullptr, the currently bound FBO will be used
        */
        void render(RenderContext* pRenderCtx, Camera* pCamera, const Fbo::SharedPtr& pTarget = nullptr);

        /** Set the sampler used to render the sky box.
        */
        void setSampler(Sampler::SharedPtr pSampler);

        /** Get the sky box texture.
        */
        Texture::SharedPtr getTexture() const;

        /** Get the sampler used to render the sky box.
        */
        Sampler::SharedPtr getSampler() const;

        void setScale(float scale) { mScale = scale; }
        float getScale() const { return mScale; }

        /** Called once before compilation. Describes I/O requirements of the pass.
        The requirements can't change after the graph is compiled. If the IO requests are dynamic, you'll need to trigger compilation of the render-graph yourself.
        */
        virtual void reflect(RenderPassReflection& reflector) const override;

        /** Executes the pass.
        */
        virtual void execute(RenderContext* pRenderContext, const RenderData* pData) override;

        virtual void setScene(const std::shared_ptr<Scene>& pScene) override { mpScene = pScene; }
    private:
        SkyBox();
        bool createResources(Texture::SharedPtr& pTexture, Sampler::SharedPtr pSampler, bool renderStereo);

        size_t mMatOffset;
        size_t mScaleOffset;

        float mScale = 1;
        Model::SharedPtr mpCubeModel;
        Texture::SharedPtr mpTexture;

        GraphicsProgram::SharedPtr mpProgram;
        GraphicsVars::SharedPtr mpVars;
        GraphicsState::SharedPtr mpState;
        Fbo::SharedPtr mpFbo;
        std::shared_ptr<Scene> mpScene;

        struct
        {
            ProgramReflection::BindLocation perFrameCB;
            ProgramReflection::BindLocation sampler;
            ProgramReflection::BindLocation texture;
        } mBindLocations;
    };
}