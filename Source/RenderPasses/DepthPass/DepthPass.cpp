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
#include "DepthPass.h"
#include "RenderGraph/RenderPassLibrary.h"

const RenderPass::Info DepthPass::kInfo { "DepthPass", "Creates a depth-buffer using the scene's active camera." };

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(DepthPass::kInfo, DepthPass::create);
}

namespace
{
    const std::string kProgramFile = "RenderPasses/DepthPass/DepthPass.3d.slang";

    const std::string kDepth = "depth";
    const std::string kDepthFormat = "depthFormat";
    const std::string kUseAlphaTest = "useAlphaTest";
}

void DepthPass::parseDictionary(const Dictionary& dict)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kDepthFormat) setDepthBufferFormat(value);
        else if (key == kUseAlphaTest) mUseAlphaTest = value;
        else logWarning("Unknown field '{}' in a DepthPass dictionary.", key);
    }
}

Dictionary DepthPass::getScriptingDictionary()
{
    Dictionary d;
    d[kDepthFormat] = mDepthFormat;
    d[kUseAlphaTest] = mUseAlphaTest;
    return d;
}

DepthPass::SharedPtr DepthPass::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new DepthPass(dict));
}

DepthPass::DepthPass(const Dictionary& dict)
    : RenderPass(kInfo)
{
    mpState = GraphicsState::create();
    mpFbo = Fbo::create();

    parseDictionary(dict);
}

RenderPassReflection DepthPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput(kDepth, "Depth-buffer").bindFlags(Resource::BindFlags::DepthStencil).format(mDepthFormat).texture2D(mOutputSize.x, mOutputSize.y);
    return reflector;
}

void DepthPass::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;
    mpVars = nullptr;

    if (mpScene)
    {
        auto defines = mpScene->getSceneDefines();
        defines.add("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");

        Program::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kProgramFile).vsEntry("vsMain").psEntry("psMain");
        desc.addTypeConformances(mpScene->getTypeConformances());
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::create(desc, defines);

        mpVars = GraphicsVars::create(pProgram->getReflector());
        mpState->setProgram(pProgram);
    }
}

void DepthPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pDepth = renderData.getTexture(kDepth);
    mpFbo->attachDepthStencilTarget(pDepth);

    mpState->setFbo(mpFbo);
    pRenderContext->clearDsv(pDepth->getDSV().get(), 1, 0);

    if (mpScene)
    {
        mpState->getProgram()->addDefine("USE_ALPHA_TEST", mUseAlphaTest ? "1" : "0");
        mpScene->rasterize(pRenderContext, mpState.get(), mpVars.get(), mCullMode);
    }
}

DepthPass& DepthPass::setDepthBufferFormat(ResourceFormat format)
{
    if (isDepthStencilFormat(format) == false)
    {
        logWarning("DepthPass buffer format must be a depth-stencil format");
    }
    else
    {
        mDepthFormat = format;
        requestRecompile();
    }
    return *this;
}

DepthPass& DepthPass::setDepthStencilState(const DepthStencilState::SharedPtr& pDsState)
{
    mpState->setDepthStencilState(pDsState);
    return *this;
}

void DepthPass::setOutputSize(const uint2& outputSize)
{
    if (outputSize != mOutputSize)
    {
        mOutputSize = outputSize;
        requestRecompile();
    }
}

void DepthPass::setAlphaTest(bool useAlphaTest)
{
    mUseAlphaTest = useAlphaTest;
}

static const Gui::DropdownList kDepthFormats =
{
    { (uint32_t)ResourceFormat::D16Unorm, "D16Unorm"},
    { (uint32_t)ResourceFormat::D32Float, "D32Float" },
    { (uint32_t)ResourceFormat::D24UnormS8, "D24UnormS8" },
    { (uint32_t)ResourceFormat::D32FloatS8X24, "D32FloatS8X24" },
};

void DepthPass::renderUI(Gui::Widgets& widget)
{
    uint32_t depthFormat = (uint32_t)mDepthFormat;

    if (widget.dropdown("Buffer Format", kDepthFormats, depthFormat)) setDepthBufferFormat(ResourceFormat(depthFormat));
}
