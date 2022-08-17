/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "SkyBox.h"
#include "RenderGraph/RenderPassLibrary.h"
#include <glm/gtx/transform.hpp>

const RenderPass::Info SkyBox::kInfo { "SkyBox", "Render an environment map. The map can be provided by the user or taken from a scene." };

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

static void regSkyBox(pybind11::module& m)
{
    pybind11::class_<SkyBox, RenderPass, SkyBox::SharedPtr> pass(m, "SkyBox");
    pass.def_property("scale", &SkyBox::getScale, &SkyBox::setScale);
    pass.def_property("filter", &SkyBox::getFilter, &SkyBox::setFilter);
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(SkyBox::kInfo, SkyBox::create);
    ScriptBindings::registerBinding(regSkyBox);
}

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
    const std::string kFilter = "filter";
}

SkyBox::SkyBox()
    : RenderPass(kInfo)
{
    mpCubeScene = Scene::create("cube.obj");

    mpProgram = GraphicsProgram::createFromFile("RenderPasses/SkyBox/SkyBox.3d.slang", "vsMain", "psMain");
    mpProgram->addDefines(mpCubeScene->getSceneDefines());
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
    rastDesc.setCullMode(RasterizerState::CullMode::None).setDepthClamp(true);
    mpRsState = RasterizerState::create(rastDesc);

    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthWriteMask(false).setDepthFunc(DepthStencilState::Func::LessEqual);
    mpState->setDepthStencilState(DepthStencilState::create(dsDesc));
    mpState->setProgram(mpProgram);

    setFilter((uint32_t)mFilter);
}

SkyBox::SharedPtr SkyBox::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    SharedPtr pSkyBox = SharedPtr(new SkyBox());
    for (const auto& [key, value] : dict)
    {
        if (key == kTexName) pSkyBox->mTexPath = value.operator std::filesystem::path();
        else if (key == kLoadAsSrgb) pSkyBox->mLoadSrgb = value;
        else if (key == kFilter) pSkyBox->setFilter(value);
        else logWarning("Unknown field '{}' in a SkyBox dictionary.", key);
    }

    std::shared_ptr<Texture> pTexture;
    if (!pSkyBox->mTexPath.empty())
    {
        pTexture = Texture::createFromFile(pSkyBox->mTexPath, false, pSkyBox->mLoadSrgb);
        if (pTexture == nullptr) throw RuntimeError("SkyBox: Failed to load skybox texture '{}'", pSkyBox->mTexPath);
        pSkyBox->setTexture(pTexture);
    }
    return pSkyBox;
}

Dictionary SkyBox::getScriptingDictionary()
{
    Dictionary dict;
    dict[kTexName] = mTexPath;
    dict[kLoadAsSrgb] = mLoadSrgb;
    dict[kFilter] = mFilter;
    return dict;
}

RenderPassReflection SkyBox::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput(kTarget, "Color buffer").format(ResourceFormat::RGBA32Float);
    auto& depthField = reflector.addInputOutput(kDepth, "Depth-buffer. Should be pre-initialized or cleared before calling the pass").bindFlags(Resource::BindFlags::DepthStencil);
    return reflector;
}

void SkyBox::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpFbo->attachColorTarget(renderData.getTexture(kTarget), 0);
    mpFbo->attachDepthStencilTarget(renderData.getTexture(kDepth));

    pRenderContext->clearRtv(mpFbo->getRenderTargetView(0).get(), float4(0));

    if (!mpScene) return;

    const auto& pEnvMap = mpScene->getEnvMap();
    mpProgram->addDefine("_USE_ENV_MAP", pEnvMap ? "1" : "0");
    if (pEnvMap) pEnvMap->setShaderData(mpVars["PerFrameCB"]["gEnvMap"]);

    rmcv::mat4 world = rmcv::translate(mpScene->getCamera()->getPosition());
    mpVars["PerFrameCB"]["gWorld"] = world;
    mpVars["PerFrameCB"]["gScale"] = mScale;
    mpVars["PerFrameCB"]["gViewMat"] = mpScene->getCamera()->getViewMatrix();
    mpVars["PerFrameCB"]["gProjMat"] = mpScene->getCamera()->getProjMatrix();
    mpState->setFbo(mpFbo);
    mpCubeScene->rasterize(pRenderContext, mpState.get(), mpVars.get(), mpRsState, mpRsState);
}

void SkyBox::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;

    if (mpScene)
    {
        mpCubeScene->setCamera(mpScene->getCamera());
        if (mpScene->getEnvMap()) setTexture(mpScene->getEnvMap()->getEnvMap());
    }
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
    std::filesystem::path path;
    FileDialogFilterVec filters = { {"bmp"}, {"jpg"}, {"dds"}, {"png"}, {"tiff"}, {"tif"}, {"tga"} };
    if (openFileDialog(filters, path))
    {
        mpTexture = Texture::createFromFile(path, false, mLoadSrgb);
        setTexture(mpTexture);
    }
}

void SkyBox::setTexture(const Texture::SharedPtr& pTexture)
{
    mpTexture = pTexture;
    if (mpTexture)
    {
        FALCOR_ASSERT(mpTexture->getType() == Texture::Type::TextureCube || mpTexture->getType() == Texture::Type::Texture2D);
        mpProgram->addDefine("_USE_SPHERICAL_MAP", mpTexture->getType() == Texture::Type::Texture2D ? "1" : "0");
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
