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
#include "DepthStencilStateTest.h"

void DepthStencilStateTest::addTests()
{
    addTestToList<TestCreate>();
}

testing_func(DepthStencilStateTest, TestCreate)
{
    const uint32_t boolCombos = 8;
    const bool depthTest[boolCombos] = { true, false, true, true, false, true, false, false };
    const bool writeDepth[boolCombos] = { true, false, true, false, true, false, true, false };
    const bool stencilTest[boolCombos] = { true, false, false, true, true, false, false, true };
    const uint32_t numCompareFunc = 9;
    const uint32_t numFaceModes = 3;
    const uint32_t numStencilOps = 8;
    TestDesc desc;
    for (uint32_t i = 0; i < boolCombos; ++i)
    {
        desc.setDepthTest(depthTest[i]);
        desc.setDepthWriteMask(writeDepth[i]);
        desc.setStencilTest(stencilTest[i]);

        //depth comparison func
        for (uint32_t j = 0; j < numCompareFunc; ++j)
        {
            desc.setDepthFunc(static_cast<DepthStencilState::Func>(j));
            //face mode
            for (uint32_t k = 0; k < numFaceModes; ++k)
            {
                //stencil fail
                for (uint32_t x = 0; x < numStencilOps; ++x)
                {
                    //depth fail
                    for (uint32_t y = 0; y < numStencilOps; ++y)
                    {
                        //depth pass
                        for (uint32_t z = 0; z < numStencilOps; ++z)
                        {
                            desc.setStencilOp(
                                static_cast<DepthStencilState::Face>(k),
                                static_cast<DepthStencilState::StencilOp>(x),
                                static_cast<DepthStencilState::StencilOp>(y),
                                static_cast<DepthStencilState::StencilOp>(z));

                            //read mask
                            for (uint8 m = 0; m < 8; ++m)
                            {
                                desc.setStencilReadMask(1 << m);
                                //write mask
                                for (uint8 n = 0; n < 8; ++n)
                                {
                                    desc.setStencilWriteMask(1 << n);
                                    DepthStencilState::SharedPtr state = DepthStencilState::create(desc);
                                    if (!doStatesMatch(state, desc))
                                    {
                                        return test_fail("State doesn't match desc used to create it");
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

bool DepthStencilStateTest::doStatesMatch(const DepthStencilState::SharedPtr state, const TestDesc& desc)
{
    return state->isDepthTestEnabled() == desc.mDepthEnabled &&
        state->isDepthWriteEnabled() == desc.mWriteDepth &&
        state->getDepthFunc() == desc.mDepthFunc &&
        state->isStencilTestEnabled() == desc.mStencilEnabled &&
        doStencilStatesMatch(state->getStencilDesc(DepthStencilState::Face::Front), desc.mStencilFront) &&
        doStencilStatesMatch(state->getStencilDesc(DepthStencilState::Face::Back), desc.mStencilBack) &&
        state->getStencilReadMask() == desc.mStencilReadMask &&
        state->getStencilWriteMask() == desc.mStencilWriteMask;
}

bool DepthStencilStateTest::doStencilStatesMatch(const DepthStencilState::StencilDesc& a, const DepthStencilState::StencilDesc& b)
{
    return a.func == b.func &&
        a.stencilFailOp == b.stencilFailOp &&
        a.depthFailOp == b.depthFailOp &&
        a.depthStencilPassOp == b.depthStencilPassOp;
}

int main()
{
    DepthStencilStateTest dsst;
    dsst.init();
    dsst.run();
    return 0;
}