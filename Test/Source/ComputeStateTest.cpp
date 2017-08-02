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
#include "ComputeStateTest.h"
#include "TestHelper.h"


void ComputeStateTest::addTests()
{
    addTestToList<TestAll>();   
}


//  
testing_func(ComputeStateTest, TestAll)
{
    //  Get the Render Context.
    RenderContext::SharedPtr pRenderContext = gpDevice->getRenderContext();
    
    //  Create the Compute Texture.
    Texture::SharedPtr pTexture = TestHelper::createRGBA32FRWTexture(1, 1);

    //  Create the Compute Programs.
    ComputeProgram::SharedPtr pCPBlack = ComputeProgram::createFromFile("CSBlack.cs.hlsl");
    ComputeVars::SharedPtr pCVBlack = ComputeVars::create(pCPBlack->getActiveVersion()->getReflector());
    pCVBlack->setTexture("gOutput", pTexture);

    ComputeProgram::SharedPtr pCPWhite = ComputeProgram::createFromFile("CSWhite.cs.hlsl");
    ComputeVars::SharedPtr pCVWhite = ComputeVars::create(pCPWhite->getActiveVersion()->getReflector());
    pCVWhite->setTexture("gOutput", pTexture);

    //  Create the Compute State.
    ComputeState::SharedPtr pCS = ComputeState::create();

    //  Set the Compute State Program to White.
    pCS->setProgram(pCPWhite);

    //  Set the Compute State and Variables for Compute Program White
    pRenderContext->setComputeState(pCS);
    pRenderContext->setComputeVars(pCVWhite);

    //  Dispatch!
    pRenderContext->dispatch(1, 1, 1);
    vec4 outputColorWhite = *(vec4*) ((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Output Color is White.
    if (outputColorWhite != vec4(1.0, 1.0, 1.0, 1.0))
    {
        return test_fail("Output Color Not White!");
    }

    //  Set the Compute State Program to White.
    pCS->setProgram(pCPBlack);

    //  Set the Compute State and Variables for Compute Program Black
    pRenderContext->setComputeState(pCS);
    pRenderContext->setComputeVars(pCVBlack);

    //  Dispatch!
    pRenderContext->dispatch(1, 1, 1);
    vec4 outputColorBlack = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Ouput Color is Black.
    if (outputColorBlack != vec4(0.0, 0.0, 0.0, 0.0))
    {
        return test_fail("Output Color Not Black!");
    }
    
    //  Return success.
    return test_pass();
}


int main()
{
    ComputeStateTest csT;
    csT.init(true);
    csT.run();
    return 0;
}
