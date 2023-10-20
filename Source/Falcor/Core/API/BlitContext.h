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
#pragma once
#include "Sampler.h"
#include "FBO.h"
#include "ParameterBlock.h"
#include "Utils/Math/Vector.h"
#include <memory>

namespace Falcor
{
class Device;
class FullScreenPass;

struct BlitContext
{
    ref<FullScreenPass> pPass;
    ref<Fbo> pFbo;

    ref<Sampler> pLinearSampler;
    ref<Sampler> pPointSampler;
    ref<Sampler> pLinearMinSampler;
    ref<Sampler> pPointMinSampler;
    ref<Sampler> pLinearMaxSampler;
    ref<Sampler> pPointMaxSampler;

    ref<ParameterBlock> pBlitParamsBuffer;
    float2 prevSrcRectOffset = float2(0, 0);
    float2 prevSrcReftScale = float2(0, 0);

    // Variable offsets in constant buffer
    TypedShaderVarOffset offsetVarOffset;
    TypedShaderVarOffset scaleVarOffset;
    ProgramReflection::BindLocation texBindLoc;

    // Parameters for complex blit
    float4 prevComponentsTransform[4] = {float4(0), float4(0), float4(0), float4(0)};
    TypedShaderVarOffset compTransVarOffset[4];

    BlitContext(Device* pDevice);
};
} // namespace Falcor
