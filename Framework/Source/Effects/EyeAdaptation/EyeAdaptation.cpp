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
#include "Framework.h"
#include "EyeAdaptation.h"
#include "Graphics/Program/ProgramVars.h"
#include "Graphics/Camera/Camera.h"
#include "API/RenderContext.h"
#include "Utils/Gui.h"
#include "Graphics/RenderGraph/RenderPassSerializer.h"

namespace Falcor
{
    static const char* kInputOutputName = "color";
    const uint32_t kMaxKernelSize = 15;


    EyeAdaptation::SharedPtr EyeAdaptation::create()
    {
        return SharedPtr(new EyeAdaptation());
    }

    EyeAdaptation::EyeAdaptation() : RenderPass("EyeAdaptation")
    {
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap, Sampler::AddressMode::Wrap);
        mpSampler = Sampler::create(samplerDesc);

        createShader();
    }

    void EyeAdaptation::createShader()
    {
        mpBlitPass = FullScreenPass::create("Framework/Shaders/Blit.vs.slang", "Effects/EyeAdaptation.ps.slang");

        ProgramReflection::SharedConstPtr pReflector = mpBlitPass->getProgram()->getReflector();
        mpVars = GraphicsVars::create(pReflector);
        mSrcTexLoc = pReflector->getDefaultParameterBlock()->getResourceBinding("srcTex");
        mNoiseTexLoc = pReflector->getDefaultParameterBlock()->getResourceBinding("noiseTex");
        mpVars = GraphicsVars::create(mpBlitPass->getProgram()->getReflector());
        mpVars["SrcRectCB"]["gOffset"] = vec2(0.0f);
        mpVars["SrcRectCB"]["gScale"] = vec2(1.0f);
        mpVars->setSampler("gSampler", mpSampler);
    }

    EyeAdaptation::SharedPtr EyeAdaptation::deserialize(const RenderPassSerializer& serializer)
    {
        Scene::UserVariable firstVar = serializer.getValue("EyeAdaptation.");
        if (firstVar.type == Scene::UserVariable::Type::Unknown)
        {
            return create();
        }

        return create();
    }

    void EyeAdaptation::reflect(RenderPassReflection& reflector) const
    {
        reflector.addInputOutput("color");
    }

    void EyeAdaptation::serialize(RenderPassSerializer& renderPassSerializer)
    {
        // renderPassSerializer.addVariable("EyeAdaptation.", );
    }

    void EyeAdaptation::updateValuesFromCamera()
    {
        Camera::SharedPtr pCamera = mpScene->getActiveCamera();
        if (!pCamera)
        {
            logWarning("No active camera for eye adaptation");
            return;
        }

        // we can calculate the EV for the camera using this formula from unreal:
        // EV = log2( Aperture * shutterSpeed * 100 / (ISO sensity))  
        
        
    }

    void EyeAdaptation::execute(RenderContext* pRenderContext, const RenderData* pData)
    {
        if (!mpTargetFbo) mpTargetFbo = Fbo::create();

        mpTargetFbo->attachColorTarget(pData->getTexture(kInputOutputName), 0);
        execute(pRenderContext, mpTargetFbo);
    }


    void EyeAdaptation::execute(RenderContext* pRenderContext, Fbo::SharedPtr pFbo)
    {
        // obtain average frame luminance


    }

    void EyeAdaptation::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (uiGroup == nullptr || pGui->beginGroup(uiGroup))
        {


            pGui->endGroup();
        }
    }

}
