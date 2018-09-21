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
#include "Falcor.h"

using namespace Falcor;

class Particles : public Renderer
{
public:
    void onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext) override;
    void onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo) override;
    void onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height) override;
    bool onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent) override;
    void onGuiRender(SampleCallbacks* pSample, Gui* pGui) override;

private:
    enum class ExamplePixelShaders
    {
        ConstColor = 0,
        ColorInterp = 1,
        Textured = 2,
        Count 
    };

    struct GuiData
    {
        int32_t mSystemIndex = -1;
        uint32_t mPixelShaderIndex = 0;
        bool mSortSystem = false;
        int32_t mMaxParticles = 4096;
        int32_t mMaxEmitPerFrame = 512;
        Gui::DropdownList mTexDropdown;
    } mGuiData;

    struct PixelShaderData
    {
        PixelShaderData(vec4 color) { type = ExamplePixelShaders::ConstColor; colorData.color1 = color; }
        PixelShaderData(ColorInterpPsPerFrame data) { type = ExamplePixelShaders::ColorInterp; colorData = data; }
        PixelShaderData(uint32_t newTexIndex, ColorInterpPsPerFrame data) 
        {
            type = ExamplePixelShaders::Textured; 
            texIndex = newTexIndex; 
            colorData = data;
        }
       
        ExamplePixelShaders type;
        ColorInterpPsPerFrame colorData;
        uint32_t texIndex;
    };

    void createSystemGui(RenderContext* pContext, Gui* pGui);
    void editPropertiesGui(Gui* pGui);
    void updateColorInterpolation(Gui* pGui);

    std::vector<ParticleSystem::SharedPtr> mpParticleSystems;
    Camera::SharedPtr mpCamera;
    FirstPersonCameraController mpCamController;
    std::vector<PixelShaderData> mPsData;
    std::vector<Texture::SharedPtr> mpTextures;
};
