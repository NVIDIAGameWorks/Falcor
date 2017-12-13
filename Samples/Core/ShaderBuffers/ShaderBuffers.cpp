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
#include "ShaderBuffers.h"

void ShaderBuffersSample::onGuiRender()
{
     mpGui->addDirectionWidget("Light Direction", mLightData.worldDir);
     mpGui->addRgbColor("Light intensity", mLightData.intensity);
     mpGui->addRgbColor("Surface Color", mSurfaceColor);
     mpGui->addCheckBox("Count FS invocations", mCountPixelShaderInvocations);
}

Vao::SharedConstPtr ShaderBuffersSample::getVao()
{
    auto pMesh = mpModel->getMesh(0);
    auto pVao = pMesh->getVao();
    return pVao;
}

void ShaderBuffersSample::onLoad()
{
    mpCamera = Camera::create();

    // create the program
    mpProgram = GraphicsProgram::createFromFile(appendShaderExtension("ShaderBuffers.vs"), appendShaderExtension("ShaderBuffers.ps"));

    // Load the model
    mpModel = Model::createFromFile("teapot.obj");

    // Plane has only one mesh, get the VAO now
    mpVao = getVao();
    auto pMesh = mpModel->getMesh(0);
    mIndexCount = pMesh->getIndexCount();

    // Set camera parameters
    glm::vec3 center = mpModel->getCenter();
    float radius = mpModel->getRadius();

    float nearZ = 0.1f;
    float farZ = radius * 100;
    mpCamera->setDepthRange(nearZ, farZ);

    // Initialize the camera controller
    mCameraController.attachCamera(mpCamera);
    mCameraController.setModelParams(center, radius, radius * 2.5f);

    // create the uniform buffers
    mpProgramVars = GraphicsVars::create(mpProgram->getActiveVersion()->getReflector());
    mpSurfaceColorBuffer = TypedBuffer<vec3>::create(1);
    uint32_t z = 0;
    mpInvocationsBuffer = Buffer::create(sizeof(uint32_t), Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::Read, &z);
    mpProgramVars->setRawBuffer("gInvocationBuffer", mpInvocationsBuffer);
    mpProgramVars->setTypedBuffer("gSurfaceColor[1]", mpSurfaceColorBuffer);

    mpRWBuffer = StructuredBuffer::create(mpProgram, "gRWBuffer", 4);
    mpProgramVars->setStructuredBuffer("gRWBuffer", mpRWBuffer);

    // create pipeline cache
    RasterizerState::Desc rsDesc;
    rsDesc.setCullMode(RasterizerState::CullMode::Back);
    mpDefaultPipelineState->setRasterizerState(RasterizerState::create(rsDesc));

    // Depth test
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthTest(true);
    mpDefaultPipelineState->setDepthStencilState(DepthStencilState::create(dsDesc));
    mpDefaultPipelineState->setFbo(mpDefaultFBO);
    mpDefaultPipelineState->setVao(mpVao);
    mpDefaultPipelineState->setProgram(mpProgram);

    // Compute
    mpComputeProgram = ComputeProgram::createFromFile(appendShaderExtension("ShaderBuffers.cs"));
    mpComputeState = ComputeState::create();
    mpComputeState->setProgram(mpComputeProgram);

    mpComputeVars = ComputeVars::create(mpComputeProgram->getActiveVersion()->getReflector());
    mpComputeVars->setStructuredBuffer("gLightIn", StructuredBuffer::create(mpComputeProgram, "gLightIn", 2));

    mpAppendLightData = StructuredBuffer::create(mpComputeProgram, "gLightOut", 2);
    mpComputeVars->setStructuredBuffer("gLightOut", mpAppendLightData);

    initializeTesting();
}

void ShaderBuffersSample::onFrameRender()
{
    beginTestFrame();

    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    mpRenderContext->clearFbo(mpDefaultFBO.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    mCameraController.update();

    //
    // Compute
    //

    mpRenderContext->clearUAV(mpAppendLightData->getUAV().get(), uvec4(0));
    mpRenderContext->clearUAVCounter(mpAppendLightData, 0);

    // Send lights to compute shader
    mpComputeVars->getStructuredBuffer("gLightIn")[0]["vec3Val"] = mLightData.worldDir;
    mpComputeVars->getStructuredBuffer("gLightIn")[1]["vec3Val"] = mLightData.intensity;
    mpComputeVars->setStructuredBuffer("gLightOut", mpAppendLightData);

    mpRenderContext->setComputeState(mpComputeState);
    mpRenderContext->setComputeVars(mpComputeVars);

    // Compute shader passes light data through an append buffer
    mpRenderContext->dispatch(1, 1, 1);

    //
    // Render
    //

    // Bind compute output
    mpProgramVars->setStructuredBuffer("gLight[3]", mpAppendLightData);
    mpRenderContext->setGraphicsState(mpDefaultPipelineState);

    // Update uniform-buffers data
    mpProgramVars["PerFrameCB"]["gWorldMat"] = glm::mat4();
    glm::mat4 wvp = mpCamera->getViewProjMatrix();
    mpProgramVars["PerFrameCB"]["gWvpMat"] = wvp;

    mpSurfaceColorBuffer[0] = mSurfaceColor;
    mpSurfaceColorBuffer->uploadToGPU();

    // Set uniform buffers
    mpRenderContext->setGraphicsVars(mpProgramVars);
    mpRenderContext->drawIndexed(mIndexCount, 0, 0);

    // Read UAV counter from append buffer
    uint32_t* pCounter = (uint32_t*)mpAppendLightData->getUAVCounter()->map(Buffer::MapType::Read);
    std::string msg = "Light Data append buffer count: " + std::to_string(*pCounter);
    renderText(msg, vec2(600, 80));
    mpAppendLightData->getUAVCounter()->unmap();

    if(mCountPixelShaderInvocations)
    {
        // RWByteAddressBuffer
        uint32_t* pData = (uint32_t*)mpInvocationsBuffer->map(Buffer::MapType::Read);
        std::string msg = "PS was invoked " + std::to_string(*pData) + " times";
        renderText(msg, vec2(600, 100));
        mpInvocationsBuffer->unmap();

        // RWStructuredBuffer UAV Counter
        pData = (uint32_t*)mpRWBuffer->getUAVCounter()->map(Buffer::MapType::Read);
        msg = "UAV Counter counted " + std::to_string(*pData) + " times";
        renderText(msg, vec2(600, 120));
        mpRWBuffer->getUAVCounter()->unmap();

        mpRenderContext->clearUAV(mpInvocationsBuffer->getUAV().get(), uvec4(0));
        mpRenderContext->clearUAVCounter(mpRWBuffer, 0);
    }

    endTestFrame();
}

void ShaderBuffersSample::onDataReload()
{
    mpVao = getVao();
}

bool ShaderBuffersSample::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return mCameraController.onKeyEvent(keyEvent);
}

bool ShaderBuffersSample::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mCameraController.onMouseEvent(mouseEvent);
}

void ShaderBuffersSample::onResizeSwapChain()
{
    float height = (float)mpDefaultFBO->getHeight();
    float width = (float)mpDefaultFBO->getWidth();

    mpCamera->setFocalLength(60.0f);
    mpCamera->setAspectRatio(width / height);
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
#ifdef FALCOR_VK
    msgBox("Vulkan support for the features used in the Shader Buffers sample is coming soon!");
    return 0;
#endif

    ShaderBuffersSample buffersSample;
    SampleConfig config;
    config.windowDesc.title = "Shader Buffers";
    config.windowDesc.resizableWindow = true;
#ifdef _WIN32
    buffersSample.run(config);
#else
    buffersSample.run(config, (uint32_t)argc, argv);
#endif
    return 0;
}
