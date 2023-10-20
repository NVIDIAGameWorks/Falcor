/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "BlitContext.h"
#include "Core/Error.h"
#include "Core/API/Device.h"
#include "Core/Program/Program.h"
#include "Core/Pass/FullScreenPass.h"

namespace Falcor
{
BlitContext::BlitContext(Device* pDevice)
{
    FALCOR_ASSERT(pDevice);

    // Init the blit data.
    DefineList defines = {
        {"SAMPLE_COUNT", "1"},
        {"COMPLEX_BLIT", "0"},
        {"SRC_INT", "0"},
        {"DST_INT", "0"},
    };
    ProgramDesc d;
    d.addShaderLibrary("Core/API/BlitReduction.3d.slang").vsEntry("vsMain").psEntry("psMain");
    pPass = FullScreenPass::create(ref<Device>(pDevice), d, defines);
    pPass->breakStrongReferenceToDevice();
    pFbo = Fbo::create(ref<Device>(pDevice));
    pFbo->breakStrongReferenceToDevice();
    FALCOR_ASSERT(pPass && pFbo);

    pBlitParamsBuffer = pPass->getVars()->getParameterBlock("BlitParamsCB");
    offsetVarOffset = pBlitParamsBuffer->getVariableOffset("gOffset");
    scaleVarOffset = pBlitParamsBuffer->getVariableOffset("gScale");
    prevSrcRectOffset = float2(-1.0f);
    prevSrcReftScale = float2(-1.0f);

    Sampler::Desc desc;
    desc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    desc.setReductionMode(TextureReductionMode::Standard);
    desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
    pLinearSampler = pDevice->createSampler(desc);
    pLinearSampler->breakStrongReferenceToDevice();
    desc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    pPointSampler = pDevice->createSampler(desc);
    pPointSampler->breakStrongReferenceToDevice();
    // Min reductions.
    desc.setReductionMode(TextureReductionMode::Min);
    desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
    pLinearMinSampler = pDevice->createSampler(desc);
    pLinearMinSampler->breakStrongReferenceToDevice();
    desc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    pPointMinSampler = pDevice->createSampler(desc);
    pPointMinSampler->breakStrongReferenceToDevice();
    // Max reductions.
    desc.setReductionMode(TextureReductionMode::Max);
    desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
    pLinearMaxSampler = pDevice->createSampler(desc);
    pLinearMaxSampler->breakStrongReferenceToDevice();
    desc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    pPointMaxSampler = pDevice->createSampler(desc);
    pPointMaxSampler->breakStrongReferenceToDevice();

    const auto& pDefaultBlockReflection = pPass->getProgram()->getReflector()->getDefaultParameterBlock();
    texBindLoc = pDefaultBlockReflection->getResourceBinding("gTex");

    // Complex blit parameters

    compTransVarOffset[0] = pBlitParamsBuffer->getVariableOffset("gCompTransformR");
    compTransVarOffset[1] = pBlitParamsBuffer->getVariableOffset("gCompTransformG");
    compTransVarOffset[2] = pBlitParamsBuffer->getVariableOffset("gCompTransformB");
    compTransVarOffset[3] = pBlitParamsBuffer->getVariableOffset("gCompTransformA");
    prevComponentsTransform[0] = float4(1.0f, 0.0f, 0.0f, 0.0f);
    prevComponentsTransform[1] = float4(0.0f, 1.0f, 0.0f, 0.0f);
    prevComponentsTransform[2] = float4(0.0f, 0.0f, 1.0f, 0.0f);
    prevComponentsTransform[3] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    for (uint32_t i = 0; i < 4; i++)
        pBlitParamsBuffer->setVariable(compTransVarOffset[i], prevComponentsTransform[i]);
}
} // namespace Falcor
