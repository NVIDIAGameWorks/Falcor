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
#include "TestBase.h"

class BlendStateTest : public TestBase
{
private:
    class TestDesc : public BlendState::Desc
    {
        friend class BlendStateTest;
    };

    void addTests() override;
    void onInit() override {};
    register_testing_func(TestCreate)
    register_testing_func(TestRtArray)
    register_testing_func(TestBlend)

    static bool doStatesMatch(const BlendState::SharedPtr state, const TestDesc& desc);
    static vec4 simulateBlend(vec4 srcColor, vec4 dstColor, vec4 blendFactor, BlendState::Desc::RenderTargetDesc::WriteMask mask, BlendState::BlendOp rgbOp,
        BlendState::BlendOp alphaOp, BlendState::BlendFunc srcRgbFunc, BlendState::BlendFunc dstRgbFunc, BlendState::BlendFunc srcAlphaFunc, BlendState::BlendFunc dstAlphaFunc);
    static vec3 applyBlendFuncRgb(vec4 srcColor, vec4 dstColor, vec4 blendFactor, BlendState::BlendFunc func, bool src);
    static float applyBlendFuncAlpha(float srcAlpha, float dstAlpha, float blendFactorAlpha, BlendState::BlendFunc func, bool src);
};
