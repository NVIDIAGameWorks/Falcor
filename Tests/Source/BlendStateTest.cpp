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
#include "BlendStateTest.h"
#include "TestHelper.h"
#include <iostream>
#include <sstream>

void BlendStateTest::addTests()
{
    addTestToList<TestCreate>();
    addTestToList<TestRtArray>();
    //addTestToList<TestBlend>();
}

testing_func(BlendStateTest, TestCreate)
{
    const uint32_t numBlendFactors = 10u;
    const BlendState::BlendOp blendOp = BlendState::BlendOp::Add;
    const BlendState::BlendFunc blendFunc = BlendState::BlendFunc::Zero;
    TestDesc desc;
    //RT stuff doesn't need to be thoroughly tested here, tested in TestRtArray
    desc.setRenderTargetWriteMask(0, true, true, true, true);
    desc.setRtBlend(0, true);
    desc.setRtParams(0, blendOp, blendOp, blendFunc, blendFunc, blendFunc, blendFunc);
    //Blend factor
    for (uint32_t i = 0; i < numBlendFactors; ++i)
    {
        float colorR = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float colorG = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float colorB = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        float colorA = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        desc.setBlendFactor(glm::vec4(colorR, colorG, colorB, colorA));

        //Set Alpha To Coverage
        for (uint32_t j = 0; j < 2; ++j)
        {
            desc.setAlphaToCoverage(j == 0u);
            //Set Independent Blend
            for (uint32_t k = 0; k < 2; ++k)
            {
                desc.setIndependentBlend(k == 0u);
                //Create and Check blend state
                BlendState::SharedPtr blendState = BlendState::create(desc);
                if (!doStatesMatch(blendState, desc))
                {
                    return test_fail("Blend state doesn't match desc used to create");
                }
            }
        }
    }

    return test_pass();
}

testing_func(BlendStateTest, TestRtArray)
{
    const uint32_t numBlendOps = 5;
    const uint32_t numBlendFuncs = 16;
    uint32_t rtIndex = 0;

    BlendState::SharedPtr state = nullptr;
    TestDesc desc;
    //rgbop
    for (uint32_t i = 0; i < numBlendOps; ++i)
    {
        //alpha op
        for (uint32_t j = 0; j < numBlendOps; ++j)
        {
            //srcRgbFunc
            for (uint32_t k = 0; k < numBlendFuncs; ++k)
            {
                //dstRgbFunc
                for (uint32_t x = 0; x < numBlendFuncs; ++x)
                {
                    //srcAlphaFunc
                    for (uint32_t y = 0; y < numBlendFuncs; ++y)
                    {
                        //dstAlphaFunc
                        for (uint32_t z = 0; z < numBlendFuncs; ++z)
                        {
                            //RT Blend
                            for (uint32_t a = 0; a < 2; ++a)
                            {
                                bool rtBlend = a == 0u;
                                //RT writeMask
                                for (uint32_t b = 0; b < 13; ++b)
                                {
                                    bool writeRed = (b & 1) != 0u;
                                    bool writeBlue = (b & 2) != 0u;
                                    bool writeGreen = (b & 4) != 0u;
                                    bool writeAlpha = (b & 8) != 0u;

                                    if (rtIndex >= Fbo::getMaxColorTargetCount())
                                    {
                                        rtIndex = 0;
                                    }

                                    //Set all properties
                                    desc.setRtParams(rtIndex,
                                        static_cast<BlendState::BlendOp>(i),
                                        static_cast<BlendState::BlendOp>(j),
                                        static_cast<BlendState::BlendFunc>(k),
                                        static_cast<BlendState::BlendFunc>(x),
                                        static_cast<BlendState::BlendFunc>(y),
                                        static_cast<BlendState::BlendFunc>(z));
                                    desc.setRtBlend(rtIndex, rtBlend);
                                    desc.setRenderTargetWriteMask(rtIndex, writeRed, writeGreen, writeBlue, writeAlpha);
                                    //Create and check state
                                    state = BlendState::create(desc);
                                    if (!doStatesMatch(state, desc))
                                    {
                                        return test_fail("Render target desc doesn't match ones used to create");
                                    }
                                    ++rtIndex;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return test_pass();
}

testing_func(BlendStateTest, TestBlend)
{
    RenderContext::SharedPtr pCtx = gpDevice->getRenderContext();
    GraphicsState::SharedPtr pState = TestHelper::getOnePixelState(pCtx.get());
    GraphicsVars::SharedPtr pVars = GraphicsVars::create(pState->getProgram()->getActiveVersion()->getReflector());

    //All of these should test blend enabled
    BlendState::Desc blendDesc;
    blendDesc.setRtBlend(0, true);


    const uint32_t numAlphaBlendFuncs = 9;
    const BlendState::BlendFunc alphaFuncs[numAlphaBlendFuncs] = { BlendState::BlendFunc::Zero, BlendState::BlendFunc::One,
        BlendState::BlendFunc::SrcAlpha, BlendState::BlendFunc::OneMinusSrcAlpha, BlendState::BlendFunc::DstAlpha,
        BlendState::BlendFunc::OneMinusDstAlpha, BlendState::BlendFunc::SrcAlphaSaturate, BlendState::BlendFunc::BlendFactor,
        BlendState::BlendFunc::OneMinusBlendFactor };
    //Multi source blend funcs will be tested in another test
    const uint32_t numRgbBlendFuncs = static_cast<uint32_t>(BlendState::BlendFunc::Src1Color);
    const uint32_t numBlendOps = 5;

    //rgbop
    for (uint32_t i = 0; i < numBlendOps; ++i)
    {
        //alpha op
        for (uint32_t j = 0; j < numBlendOps; ++j)
        {
            //srcRgbFunc
            for (uint32_t k = 1; k < numRgbBlendFuncs; ++k)
            {
                //dstRgbFunc
                for (uint32_t x = 1; x < numRgbBlendFuncs; ++x)
                {
                    //srcAlphaFunc
                    for (uint32_t y = 1; y < numAlphaBlendFuncs; ++y)
                    {
                        //dstAlphaFunc
                        for (uint32_t z = 1; z < numAlphaBlendFuncs; ++z)
                        {
                            //writeMask
                            for (uint32_t b = 0; b < 13; ++b)
                            {
                                BlendState::Desc::RenderTargetDesc::WriteMask mask;
                                mask.writeRed = (b & 1) != 0u;
                                mask.writeBlue = (b & 2) != 0u;
                                mask.writeGreen = (b & 4) != 0u;
                                mask.writeAlpha = (b & 8) != 0u;

                                vec4 blendFactor = TestHelper::randVec4ZeroToOne();
                                vec4 srcColor = TestHelper::randVec4ZeroToOne();
                                vec4 dstColor = TestHelper::randVec4ZeroToOne();

                                BlendState::BlendOp rgbOp = static_cast<BlendState::BlendOp>(i);
                                BlendState::BlendOp alphaOp = static_cast<BlendState::BlendOp>(j);
                                BlendState::BlendFunc srcRgbFunc = static_cast<BlendState::BlendFunc>(k);
                                BlendState::BlendFunc dstRgbFunc = static_cast<BlendState::BlendFunc>(x);
                                BlendState::BlendFunc srcAlphaFunc = alphaFuncs[y];
                                BlendState::BlendFunc dstAlphaFunc = alphaFuncs[z];

                                //Set all properties
                                blendDesc.setRtParams(0, rgbOp, alphaOp, srcRgbFunc, dstRgbFunc, srcAlphaFunc, dstAlphaFunc);
                                blendDesc.setBlendFactor(blendFactor);
                                blendDesc.setRenderTargetWriteMask(0, mask.writeRed, mask.writeGreen, mask.writeBlue, mask.writeAlpha);

                                pCtx->clearFbo(pState->getFbo().get(), dstColor, 0.f, 0u, FboAttachmentType::Color);
                                BlendState::SharedPtr pBlend = BlendState::create(blendDesc);
                                pState->setBlendState(pBlend);
                                pCtx->setGraphicsState(pState);
                                pVars->getConstantBuffer("PerFrameCB")->setBlob(&srcColor, 0u, sizeof(vec4));
                                pCtx->setGraphicsVars(pVars);

                                pCtx->draw(4, 0);
                                vec4 resultingColor = *(vec4*)(pCtx->readTextureSubresource(pState->getFbo()->getColorTexture(0).get(), 0).data());
                                vec4 simulateColor = simulateBlend(srcColor, dstColor, blendFactor, mask, rgbOp, alphaOp, srcRgbFunc, dstRgbFunc, srcAlphaFunc, dstAlphaFunc);

                                if (!TestHelper::nearVec4(resultingColor, simulateColor))
                                {
                                    std::stringstream ss;
                                    ss << std::endl << "rgb op: " << (uint32_t)rgbOp << ", alpha op:" << (uint32_t)alphaOp << std::endl;
                                    ss << "srcRgbFunc: " << (uint32_t)srcRgbFunc << ", dstRgbFunc : " << (uint32_t)dstRgbFunc << ", srcAlphaFunc : " << (uint32_t)srcAlphaFunc << ", dstAlphaFunc : " << (uint32_t)dstAlphaFunc << std::endl;
                                    ss << "Masks: " << mask.writeRed << ", " << mask.writeGreen << ", " << mask.writeBlue << ", " << mask.writeAlpha << std::endl;
                                    ss << "src: " << srcColor.x << ", " << srcColor.y << ", " << srcColor.z << ", " << srcColor.w << std::endl;
                                    ss << "dst: " << dstColor.x << ", " << dstColor.y << ", " << dstColor.z << ", " << dstColor.w << std::endl;
                                    ss << "factor: " << blendFactor.x << ", " << blendFactor.y << ", " << blendFactor.z << ", " << blendFactor.w << std::endl;
                                    ss << "gpu result: " << resultingColor.x << ", " << resultingColor.y << ", " << resultingColor.z << ", " << resultingColor.w << std::endl;
                                    ss << "cpu result: " << simulateColor.x << ", " << simulateColor.y << ", " << simulateColor.z << ", " << simulateColor.w << std::endl;
                                    return test_fail(ss.str());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return test_pass();
}

bool BlendStateTest::doStatesMatch(const BlendState::SharedPtr state, const TestDesc& desc)
{
    //check settings not in the rt array
    if (state->isAlphaToCoverageEnabled() != desc.mAlphaToCoverageEnabled ||
        state->getBlendFactor() != desc.mBlendFactor ||
        state->isIndependentBlendEnabled() != desc.mEnableIndependentBlend ||
        state->getRtCount() != desc.mRtDesc.size())
    {
        return false;
    }

    //check the rt array
    for (uint32_t i = 0; i < state->getRtCount(); ++i)
    {
        BlendState::Desc::RenderTargetDesc rtDesc = state->getRtDesc(i);
        BlendState::Desc::RenderTargetDesc otherRtDesc = desc.mRtDesc[i];
        if (rtDesc.writeMask.writeRed != otherRtDesc.writeMask.writeRed ||
            rtDesc.writeMask.writeGreen != otherRtDesc.writeMask.writeGreen ||
            rtDesc.writeMask.writeBlue != otherRtDesc.writeMask.writeBlue ||
            rtDesc.writeMask.writeAlpha != otherRtDesc.writeMask.writeAlpha ||
            rtDesc.blendEnabled != otherRtDesc.blendEnabled ||
            rtDesc.rgbBlendOp != otherRtDesc.rgbBlendOp ||
            rtDesc.alphaBlendOp != otherRtDesc.alphaBlendOp ||
            rtDesc.srcRgbFunc != otherRtDesc.srcRgbFunc ||
            rtDesc.dstRgbFunc != otherRtDesc.dstRgbFunc ||
            rtDesc.srcAlphaFunc != otherRtDesc.srcAlphaFunc ||
            rtDesc.dstAlphaFunc != otherRtDesc.dstAlphaFunc)
        {
            return false;
        }

    }

    return true;
}

vec4 BlendStateTest::simulateBlend(vec4 srcColor, vec4 dstColor, vec4 blendFactor, BlendState::Desc::RenderTargetDesc::WriteMask mask,
    BlendState::BlendOp rgbOp, BlendState::BlendOp alphaOp, BlendState::BlendFunc srcRgbFunc, BlendState::BlendFunc dstRgbFunc,
    BlendState::BlendFunc srcAlphaFunc, BlendState::BlendFunc dstAlphaFunc)
{
    vec3 resultSrcColor = applyBlendFuncRgb(srcColor, dstColor, blendFactor, srcRgbFunc, true);
    vec3 resultDstColor = applyBlendFuncRgb(srcColor, dstColor, blendFactor, dstRgbFunc, false);
    float resultSrcAlpha = applyBlendFuncAlpha(srcColor.w, dstColor.w, blendFactor.w, srcAlphaFunc, true);
    float resultDstAlpha = applyBlendFuncAlpha(srcColor.w, dstColor.w, blendFactor.w, dstAlphaFunc, false);

    vec3 finalRgb = vec3();
    switch (rgbOp)
    {
    case BlendState::BlendOp::Add:
        finalRgb = resultSrcColor + resultDstColor;
        break;
    case BlendState::BlendOp::Subtract:
        finalRgb = resultSrcColor - resultDstColor;
        break;
    case BlendState::BlendOp::ReverseSubtract:
        finalRgb = resultDstColor - resultSrcColor;
        break;
    case BlendState::BlendOp::Min:
        finalRgb = vec3(min(srcColor.x, dstColor.x), min(srcColor.y, dstColor.y), min(srcColor.z, dstColor.z));
        break;
    case BlendState::BlendOp::Max:
        finalRgb = vec3(max(srcColor.x, dstColor.x), max(srcColor.y, dstColor.y), max(srcColor.z, dstColor.z));
        break;
    default:
        should_not_get_here();
    }

    float finalAlpha = 0;
    switch (alphaOp)
    {
    case BlendState::BlendOp::Add:
        finalAlpha = resultSrcAlpha + resultDstAlpha;
        break;
    case BlendState::BlendOp::Subtract:
        finalAlpha = resultSrcAlpha - resultDstAlpha;
        break;
    case BlendState::BlendOp::ReverseSubtract:
        finalAlpha = resultDstAlpha - resultSrcAlpha;
        break;
    case BlendState::BlendOp::Min:
        finalAlpha = min(srcColor.w, dstColor.w);
        break;
    case BlendState::BlendOp::Max:
        finalAlpha = max(srcColor.w, dstColor.w);
        break;
    default:
        should_not_get_here();
    }

    if (!mask.writeRed)
    {
        finalRgb.x = dstColor.x;
    }

    if (!mask.writeGreen)
    {
        finalRgb.y = dstColor.y;
    }

    if (!mask.writeBlue)
    {
        finalRgb.z = dstColor.z;
    }

    if (!mask.writeAlpha)
    {
        finalAlpha = dstColor.w;
    }

    return vec4(finalRgb.x, finalRgb.y, finalRgb.z, finalAlpha);
}

vec3 BlendStateTest::applyBlendFuncRgb(vec4 srcColor, vec4 dstColor, vec4 blendFactor, BlendState::BlendFunc func, bool src)
{
    vec3 result;
    if (src)
    {
        result = vec3(srcColor.x, srcColor.y, srcColor.z);
    }
    else
    {
        result = vec3(dstColor.x, dstColor.y, dstColor.z);
    }

    vec3 oneVec3 = vec3(1, 1, 1);
    vec3 srcVec3 = vec3(srcColor.x, srcColor.y, srcColor.z);
    vec3 srcVec3a = vec3(srcColor.w, srcColor.w, srcColor.w);
    vec3 dstVec3 = vec3(dstColor.x, dstColor.y, dstColor.z);
    vec3 dstVec3a = vec3(dstColor.w, dstColor.w, dstColor.w);
    vec3 blendVec3 = vec3(blendFactor.x, blendFactor.y, blendFactor.z);
    switch (func)
    {
    case BlendState::BlendFunc::Zero:
        return result * vec3(0, 0, 0);
    case BlendState::BlendFunc::One:
        return result;
    case BlendState::BlendFunc::SrcColor:
        return result * srcVec3;
    case BlendState::BlendFunc::OneMinusSrcColor:
        return result * (oneVec3 - srcVec3);
    case BlendState::BlendFunc::DstColor:
        return result * dstVec3;
    case BlendState::BlendFunc::OneMinusDstColor:
        return result * (oneVec3 - dstVec3);
    case BlendState::BlendFunc::SrcAlpha:
        return result * srcVec3a;
    case BlendState::BlendFunc::OneMinusSrcAlpha:
        return result * (oneVec3 - srcVec3a);
    case BlendState::BlendFunc::DstAlpha:
        return result * dstVec3a;
    case BlendState::BlendFunc::OneMinusDstAlpha:
        return result * (oneVec3 - dstVec3a);
    case BlendState::BlendFunc::BlendFactor:
        return result * blendVec3;
    case BlendState::BlendFunc::OneMinusBlendFactor:
        return result * (oneVec3 - blendVec3);
    case BlendState::BlendFunc::SrcAlphaSaturate:
    {
        float val = min(srcColor.w, 1 - dstColor.w);
        return result * vec3(val, val, val);
    }
    //Not sure about any of these
    case BlendState::BlendFunc::Src1Color:
    case BlendState::BlendFunc::OneMinusSrc1Color:
    case BlendState::BlendFunc::Src1Alpha:
    case::BlendState::BlendFunc::OneMinusSrc1Alpha:
        return result;
    default:
        should_not_get_here();
        return result;
    }
}

float BlendStateTest::applyBlendFuncAlpha(float srcAlpha, float dstAlpha, float blendFactorAlpha, BlendState::BlendFunc func, bool src)
{
    float result;
    if (src)
    {
        result = srcAlpha;
    }
    else
    {
        result = dstAlpha;
    }

    switch (func)
    {
    case BlendState::BlendFunc::Zero:
        return 0;
    case BlendState::BlendFunc::One:
        return result;
    case BlendState::BlendFunc::Src1Color:
    case BlendState::BlendFunc::OneMinusSrc1Color:
    case BlendState::BlendFunc::SrcColor:
    case BlendState::BlendFunc::OneMinusSrcColor:
    case BlendState::BlendFunc::DstColor:
    case BlendState::BlendFunc::OneMinusDstColor:
        throw std::exception("Color Blend funcs not supported on alpha");
        return result;
    case BlendState::BlendFunc::SrcAlpha:
        return result * srcAlpha;
    case BlendState::BlendFunc::OneMinusSrcAlpha:
        return result * (1 - srcAlpha);
    case BlendState::BlendFunc::DstAlpha:
        return result * dstAlpha;
    case BlendState::BlendFunc::OneMinusDstAlpha:
        return result * (1 - dstAlpha);
    case BlendState::BlendFunc::BlendFactor:
        return result * blendFactorAlpha;
    case BlendState::BlendFunc::OneMinusBlendFactor:
        return result * (1 - blendFactorAlpha);
    case BlendState::BlendFunc::SrcAlphaSaturate:
        return result;
        //Not sure about any of these
    case BlendState::BlendFunc::Src1Alpha:
        return result;
    case::BlendState::BlendFunc::OneMinusSrc1Alpha:
        return result;
    default:
        should_not_get_here();
        return result;
    }
}

int main()
{
    BlendStateTest bst;
    bst.init(true);
    bst.run();
    return 0;
}
