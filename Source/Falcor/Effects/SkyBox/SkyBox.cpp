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
#include "stdafx.h"
#include "SkyBox.h"
#include "glm/gtx/transform.hpp"

namespace Falcor
{
    const char* SkyBox::kDesc = "Render an environment-map. The map can be provided by the user or taken from a scene";

    namespace
    {
        const Gui::DropdownList kFilterList =
        {
            { (uint32_t)Sampler::Filter::Linear, "Linear" },
            { (uint32_t)Sampler::Filter::Point, "Point" },
        };

        const std::string kTarget = "target";
        const std::string kDepth = "depth";

        // Dictionary keys
        const std::string kTexName = "texName";
        const std::string kLoadAsSrgb = "loadAsSrgb";
        const std::string kRenderStereo = "renderStereo";
        const std::string kFilter = "filter";
    }

    SkyBox::SkyBox()
    {
        mpCubeScene = Scene::create("Effects/cube.obj");
        if (mpCubeScene == nullptr) throw std::runtime_error("SkyBox::SkyBox - Failed to load cube model");

        mpProgram = GraphicsProgram::createFromFile("Effects/SkyBox.slang", "vs", "ps");
        mpVars = GraphicsVars::create(mpProgram->getReflector());
        mpFbo = Fbo::create();

        // Create state
        mpState = GraphicsState::create();
        BlendState::Desc blendDesc;
        for (uint32_t i = 1; i < Fbo::getMaxColorTargetCount(); i++) blendDesc.setRenderTargetWriteMask(i, false, false, false, false);
        blendDesc.setIndependentBlend(true);
        mpState->setBlendState(BlendState::create(blendDesc));

        // Create the rasterizer state
        RasterizerState::Desc rastDesc;
        rastDesc.setCullMode(RasterizerState::CullMode::Front).setDepthClamp(true);
        mpState->setRasterizerState(RasterizerState::create(rastDesc));

        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthWriteMask(false).setDepthFunc(DepthStencilState::Func::LessEqual);
        mpState->setDepthStencilState(DepthStencilState::create(dsDesc));
        mpState->setProgram(mpProgram);

        setFilter((uint32_t)mFilter);
    }

    SkyBox::SharedPtr SkyBox::create(RenderContext* pRenderContext, const Dictionary& dict)
    {
        SharedPtr pSkyBox = SharedPtr(new SkyBox());
        for (const auto& v : dict)
        {
            if (v.key() == kTexName)
            {
                std::string name = v.val();
                pSkyBox->mTexName = name;
            }
            if (v.key() == kLoadAsSrgb) pSkyBox->mLoadSrgb = v.val();
            if (v.key() == kFilter)	pSkyBox->setFilter((uint32_t)v.val());
            if (v.key() == kRenderStereo) pSkyBox->mRenderStereo = v.val();
            else logWarning("Unknown field '" + v.key() + "' in a SkyBox dictionary");
        }

        std::shared_ptr<Texture> pTexture;
        if (pSkyBox->mTexName.size() != 0)
        {
            pTexture = Texture::createFromFile(pSkyBox->mTexName, false, pSkyBox->mLoadSrgb);
            if (pTexture == nullptr) throw std::runtime_error("SkyBox::create - Error creating texture from file");
            pSkyBox->setTexture(pTexture);
        }
        return pSkyBox;
    }

    Dictionary SkyBox::getScriptingDictionary()
    {
        Dictionary dict;
        dict[kTexName] = mTexName;
        dict[kLoadAsSrgb] = mLoadSrgb;
        dict[kFilter] = mFilter;
        dict[kRenderStereo] = mRenderStereo;
        return dict;
    }

    RenderPassReflection SkyBox::reflect(const CompileData& compileData)
    {
        RenderPassReflection reflector;
        reflector.addOutput(kTarget, "Color buffer").format(ResourceFormat::RGBA32Float);
        auto& depthField = reflector.addInputOutput(kDepth, "Depth-buffer. Should be pre-initialized or cleared before calling the pass").bindFlags(Resource::BindFlags::DepthStencil);
        return reflector;
    }

    void SkyBox::compile(RenderContext* pContext, const CompileData& compileData)
    {
        // Create the program
        Program::DefineList defines;
        if (mRenderStereo) defines.add("_SINGLE_PASS_STEREO");
        mpProgram->addDefines(defines);
    }

    void SkyBox::execute(RenderContext* pRenderContext, const RenderData& renderData)
    {
        if (!mpScene) return;

        mpFbo->attachColorTarget(renderData[kTarget]->asTexture(), 0);
        mpFbo->attachDepthStencilTarget(renderData[kDepth]->asTexture());

        pRenderContext->clearRtv(mpFbo->getRenderTargetView(0).get(), vec4(0));

        glm::mat4 world = glm::translate(mpScene->getCamera()->getPosition());
        mpVars["PerFrameCB"]["gWorld"] = world;
        mpVars["PerFrameCB"]["gScale"] = mScale;
        mpVars["PerFrameCB"]["gViewMat"] = mpScene->getCamera()->getViewMatrix();
        mpVars["PerFrameCB"]["gProjMat"] = mpScene->getCamera()->getProjMatrix();
        mpState->setFbo(mpFbo);
        mpCubeScene->render(pRenderContext, mpState.get(), mpVars.get(), Scene::RenderFlags::UserRasterizerState);
    }

    void SkyBox::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
    {
        mpScene = pScene;

        if (mpScene && mpScene->getEnvironmentMap()) setTexture(mpScene->getEnvironmentMap());
        if (mpScene) mpCubeScene->setCamera(mpScene->getCamera());
    }

    void SkyBox::renderUI(Gui::Widgets& widget)
    {
        float scale = mScale;
        if (widget.var("Scale", scale, 0.f)) setScale(scale);

        if (widget.button("Load Image")) { loadImage(); }

        uint32_t filter = (uint32_t)mFilter;
        if (widget.dropdown("Filter", kFilterList, filter)) setFilter(filter);
    }

    void SkyBox::loadImage()
    {
        std::string filename;
        FileDialogFilterVec filters = { {"bmp"}, {"jpg"}, {"dds"}, {"png"}, {"tiff"}, {"tif"}, {"tga"} };
        if (openFileDialog(filters, filename))
        {
            mpTexture = Texture::createFromFile(filename, false, mLoadSrgb);
            setTexture(mpTexture);
        }
    }

    void SkyBox::setTexture(const Texture::SharedPtr& pTexture)
    {
        mpTexture = pTexture;
        if (mpTexture)
        {
            assert(mpTexture->getType() == Texture::Type::TextureCube || mpTexture->getType() == Texture::Type::Texture2D);
            (mpTexture->getType() == Texture::Type::Texture2D) ? mpProgram->addDefine("_SPHERICAL_MAP") : mpProgram->removeDefine("_SPHERICAL_MAP");
        }
        mpVars["gTexture"] = mpTexture;
    }

    void SkyBox::setFilter(uint32_t filter)
    {
        mFilter = (Sampler::Filter)filter;
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(mFilter, mFilter, mFilter);
        mpSampler = Sampler::create(samplerDesc);
        mpVars["gSampler"] = mpSampler;
    }

    SCRIPT_BINDING(SkyBox)
    {
        auto c = m.regClass(SkyBox);
        c.func_("scale", &SkyBox::setScale);
        c.func_("scale", &SkyBox::getScale);
        c.func_("filter", &SkyBox::setFilter);
        c.func_("filter", &SkyBox::getFilter);
    }
}
