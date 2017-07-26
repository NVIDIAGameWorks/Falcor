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
#include "RasterizerStateTest.h"
#include <iostream>
#include <limits>

void RasterizerStateTest::addTests()
{
    addTestToList<TestCreate>();
}

testing_func(RasterizerStateTest, TestCreate)
{
    const uint32_t numCullModes = 3u;
    const uint32_t numFillModes = 2u;
    const uint32_t numRandomDepthBiases = 5u;
    
    TestDesc desc;
    //Cull mode
    for (uint32_t i = 0; i < numCullModes; ++i)
    {
        desc.setCullMode(static_cast<RasterizerState::CullMode>(i));
        //Fill mode
        for (uint32_t j = 0; j < numFillModes; ++j)
        {
            desc.setFillMode(static_cast<RasterizerState::FillMode>(j));
            //Depth Bias
            for (uint k = 0; k < numRandomDepthBiases; ++k)
            {
                float randFloat = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                int32_t randInt = static_cast<int>(rand());
                desc.setDepthBias(randInt, randFloat);

                //Boolean properties
                //Front Counter CW
                for (uint32_t a = 0; a < 2; ++a)
                {
                    desc.setFrontCounterCW(a == 0u);
                    //Depth Clamp
                    for (uint32_t b = 0; b < 2; ++b)
                    {
                        desc.setDepthClamp(b == 0u);
                        //Line Anti Aliasing
                        for (uint32_t c = 0; c < 2; ++c)
                        {
                            desc.setLineAntiAliasing(c == 0u);
                            //scissor test
                            for (uint32_t d = 0; d < 2; ++d)
                            {
                                desc.setScissorTest(d == 0);
                                //Conservative Rasterization
                                for (uint32_t e = 0; e < 2; ++e)
                                {
                                    desc.setConservativeRasterization(e == 0u);
                                    //forced sample count
                                    for (uint32_t f = 0; f < 2; ++f)
                                    {
                                        desc.setForcedSampleCount(f == 0u);
                                        //Create and check  state
                                        RasterizerState::SharedPtr state = RasterizerState::create(desc);
                                        if (!doStatesMatch(state, desc))
                                        {
                                            return test_fail("Rasterizer state doesn't match desc used to create it");
                                        }
                                    }
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

bool RasterizerStateTest::doStatesMatch(const RasterizerState::SharedPtr state, const TestDesc& desc)
{
    return state->getCullMode() == desc.mCullMode &&
        state->getFillMode() == desc.mFillMode &&
        state->isFrontCounterCW() == desc.mIsFrontCcw &&
        state->getSlopeScaledDepthBias() == desc.mSlopeScaledDepthBias &&
        state->getDepthBias() == desc.mDepthBias &&
        state->isDepthClampEnabled() == desc.mClampDepth &&
        state->isScissorTestEnabled() == desc.mScissorEnabled &&
        state->isLineAntiAliasingEnabled() == desc.mEnableLinesAA &&
        state->isConservativeRasterizationEnabled() == desc.mConservativeRaster &&
        state->getForcedSampleCount() == desc.mForcedSampleCount;
}

int main()
{
    RasterizerStateTest rst;
    rst.init();
    rst.run();
    return 0;
}