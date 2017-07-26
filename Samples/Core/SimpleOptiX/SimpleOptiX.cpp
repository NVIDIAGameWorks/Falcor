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
#include "SimpleOptiX.h"
#include "Graphics/Scene/SceneExporter.h"

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::ivec4;
using std::string;

SimpleOptix::SimpleOptix()
{
}

SimpleOptix::~SimpleOptix()
{
}

void SimpleOptix::onShutdown() 
{
	/* Destruction order is important: context should be destroyed the last */
	mRTRaygenRtn.reset();
	mRTShadingRtn.reset();
    mRTContext.reset();
}

void SimpleOptix::loadSceneFromFile(const std::string filename)
{
    mpScene = Scene::loadFromFile(filename, Model::GenerateTangentSpace);
    if(mpScene)
    {
        mpRenderer = SceneRenderer::create(mpScene);
        uint32_t width = mpDefaultFBO->getWidth();
        uint32_t height = mpDefaultFBO->getHeight();
        for(uint32_t i = 0; i < mpScene->getCameraCount(); ++i)
        {
            if(Camera::SharedPtr camera = mpScene->getCamera(i))
            {
                camera->setAspectRatio(float(width) / float(height));
            }
        }

        // Read lighting rig
        auto dirlightDir = mpScene->getUserVariable("dirlight_dir");
        if(dirlightDir.type == Scene::UserVariable::Type::Vec3)
            mpDirLight->setWorldDirection(normalize(dirlightDir.vec3));
        auto dirlightColor = mpScene->getUserVariable("dirlight_color");
        if(dirlightColor.type == Scene::UserVariable::Type::Vec3)
            mpDirLight->setIntensity(dirlightColor.vec3);

        // Convert all paths to camera paths
        if(mpScene->getActiveCamera())
        {
            for(uint32_t pi = 0; pi < mpScene->getPathCount(); pi++)
            {
                mpScene->getPath(pi)->setInterpolationMode(ObjectPath::Interpolation::CubicSpline);
                mpScene->getPath(pi)->attachObject(mpScene->getActiveCamera());
            }
        }

        // Set texture sampler
        for(uint32_t i = 0; i < mpScene->getModelCount(); ++i)
            mpScene->getModel(i)->bindSamplerToMaterials(mpLinearSampler);

        // Update scene-dependent parameters
        mCurrentTime = 0.f;
        mFrameCount = 0;

        mRTContext->newScene(mpScene, mRTShadingRtn);
    }
}

void SimpleOptix::loadScene()
{
    string filename;
    if(openFileDialog("Scene files\0*.fscene\0\0", filename))
    {
        loadSceneFromFile(filename);
    }
}

void SimpleOptix::initUI()
{
    // Load model group
    mpGui->addButton("Load Scene", [](void* pUserData) { ((SimpleOptix*)(pUserData))->loadScene(); }, this);

    mpDirLight->setUiElements(mpGui.get(), "Directional Light");
    mpPointLight->setUiElements(mpGui.get(), "Point Light");

    mpGui->addCheckBox("Ray trace", &mRaytrace);
    mpGui->addCheckBox("Supersample", &mSupersample);
    mpGui->addIntVar("Bounces", &mBounces, "", 0, 2);
    mpGui->addFloatVarWithCallback("Exposure", [](const void* pVal, void* pUserData) { ((SimpleOptix*)(pUserData))->mExposure = exp2f(*(float*)pVal); ((SimpleOptix*)(pUserData))->mFrameCount = 0; }, [](void* pVal, void* pUserData) { *((float*)pVal) = log2f(((SimpleOptix*)(pUserData))->mExposure); }, this, "", -20.f, 30.f, 0.2f);
    
    uint32_t barSize[2];
    mpGui->getSize(barSize);
    barSize[0] += 50;
    barSize[1] += 100;
    mpGui->setSize(barSize[0], barSize[1]);
}

void SimpleOptix::onLoad()
{
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create ray tracing routines
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    assert(!mRTContext);
    mRTContext = RTContext::create();

    mRTRaygenRtn = mRTContext->createRoutine("camera", {{0, "pinhole"}});
    mRTShadingRtn = mRTContext->createRoutine("simple_shader", {{0, "closest_hit_radiance"}});
    
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // Create raster routines
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear).setMaxAnisotropy(8);
    mpLinearSampler = Sampler::create(samplerDesc);

    // Load raster routines
    mpRasterShader = Program::createFromFile("", "Raster.fs");
    mpPerFrameCB = UniformBuffer::create(mpRasterShader, "PerFrameCB");

    // create rasterizer state
    RasterizerState::Desc solidDesc;
    solidDesc.setCullMode(RasterizerState::CullMode::None);
    mpCullRastState = RasterizerState::create(solidDesc);

    // Depth test
    DepthStencilState::Desc dsDesc;
    mpDepthTestDS = DepthStencilState::create(dsDesc);

    // Lighting
    mpPointLight = PointLight::create();
    mpDirLight = DirectionalLight::create();
    mpDirLight->setWorldDirection(vec3(-0.5f, -0.2f, -1.0f));

    // Load default model
    loadSceneFromFile("Scenes//bumpyplane.fscene");

    initUI();
}

void SimpleOptix::onFrameRender()
{
    uint32_t width = mpDefaultFBO->getWidth();
    uint32_t height = mpDefaultFBO->getHeight();

    const vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    mpDefaultFBO->clear(clearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpScene == nullptr)
    {
        return;
    }

    if(mpRenderer->update(mCurrentTime))
    {
        mFrameCount = 0;
    }

    if(mRaytrace)
    {
        // Set user variables
        mRTContext->set("gSupersample", mSupersample ? 1 : 0);
        mRTContext->set("gBounces", mBounces);
        mRTContext->set("gFrameNumber", (float)mFrameCount);
        mRTContext->set("gExposure", mExposure);
        
        // Set lights
        mRTContext->setLights( { mpDirLight.get(), mpPointLight.get() } );

        mRTContext->render(mpTargetFBO, mpScene->getActiveCamera(), mRTRaygenRtn);
        mpRenderContext->blitFbo(mpTargetFBO.get(), mpDefaultFBO.get(), ivec4(0, 0, width, height), ivec4(0, 0, width, height));
        mFrameCount++;
    }
    else
    {
        mFrameCount = 0;
        // Set render state
        mpRenderContext->setRasterizerState(mpCullRastState);
        mpRenderContext->setDepthStencilState(mpDepthTestDS, 0);
        mpDirLight->setIntoUniformBuffer(mpPerFrameCB.get(), "gDirLight");
        mpPointLight->setIntoUniformBuffer(mpPerFrameCB.get(), "gPointLight");
        mpRenderContext->setUniformBuffer(0, mpPerFrameCB);

        mpRenderer->renderScene(mpRenderContext.get(), mpRasterShader.get());
    }
    
    renderText(getGlobalSampleMessage(false), vec2(10, 10));
}

bool SimpleOptix::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if(mpRenderer && mpRenderer->onKeyEvent(keyEvent))
    {
        mFrameCount = 0;
        return true;
    }

    return false;
}

bool SimpleOptix::onMouseEvent(const MouseEvent& mouseEvent)
{
    if(mpRenderer && mpRenderer->onMouseEvent(mouseEvent))
    {
        mFrameCount = 0;
        return true;
    }
    return false;
}

void SimpleOptix::onDataReload()
{
    mRTContext->reloadRoutine(mRTRaygenRtn);
    mRTContext->reloadRoutine(mRTShadingRtn);
    mFrameCount = 0;
}

void SimpleOptix::onResizeSwapChain()
{
    uint32_t width = mpDefaultFBO->getWidth();
    uint32_t height = mpDefaultFBO->getHeight();

	RenderContext::Viewport vp;
	vp.width = float(width);
	vp.height = float(height);
	mpRenderContext->setViewport(0, vp);

    // Create auxiliary FBO
    Falcor::ResourceFormat FBFormat[1] = { Falcor::ResourceFormat::RGBA32Float };
    mpTargetFBO = FboHelper::create2D(width, height, FBFormat);

    mFrameCount = 0;
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    SimpleOptix sample;
    SampleConfig config;
    config.windowDesc.swapChainDesc.width = 1280;
    config.windowDesc.swapChainDesc.height = 720;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.title = "Ray tracing sample";
    sample.run(config);
}
