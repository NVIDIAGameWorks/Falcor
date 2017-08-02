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
#include "ProgramTest.h"
#include "TestHelper.h"

void ProgramTest::addTests()
{
    //  Test Compute Program.
    addTestToList<TestComputeProgramCreate>();
    addTestToList<TestComputeProgramAddRemoveDefines>();
    addTestToList<TestComputeProgramReloadPrograms>();

    //  Test Graphics Program.
    addTestToList<TestGraphicsProgramCreate>();
    addTestToList<TestGraphicsProgramAddRemoveDefines>();
    addTestToList<TestGraphicsProgramReloadPrograms>();
}

//  Test the Compute Program Class.
testing_func(ProgramTest, TestComputeProgramCreate)
{
    //  Get the Render Context, and create the Compute Texture.
    RenderContext::SharedPtr pRenderContext = gpDevice->getRenderContext();
    Texture::SharedPtr pTexture = TestHelper::createRGBA32FRWTexture(1, 1);

    //  Create the Compute Program, and create the Compute State.
    ComputeProgram::SharedPtr pCPBase = ComputeProgram::createFromFile("CSBase.cs.hlsl");
    ComputeVars::SharedPtr pCVBase = ComputeVars::create(pCPBase->getActiveVersion()->getReflector());
    ComputeState::SharedPtr pCSBase = ComputeState::create();
    

    //  Set the Compute State Program, and set the Compute State and Variables for Compute Program
    pCVBase->setTexture("gOutput", pTexture);
    pCSBase->setProgram(pCPBase);
    pRenderContext->setComputeState(pCSBase);
    pRenderContext->setComputeVars(pCVBase);

    //  Dispatch!
    pRenderContext->dispatch(1, 1, 1);
    vec4 outputColorBase = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Output Color is Black.
    if (outputColorBase != vec4(0.0, 0.0, 0.0, 0.0))
    {
        return test_fail("Output Color Not Black!");
    }
    
    //
    //  Create the Compute Program with White.
    Program::DefineList defsW;
    defsW.add("_USE_WHITE", "");

    //  Create the Compute Program.
    ComputeProgram::SharedPtr pCPWhite = ComputeProgram::createFromFile("CSBase.cs.hlsl", defsW);
    ComputeVars::SharedPtr pCVWhite = ComputeVars::create(pCPWhite->getActiveVersion()->getReflector());
    ComputeState::SharedPtr pCSWhite = ComputeState::create();

    //
    pCVWhite->setTexture("gOutput", pTexture);
    pCSWhite->setProgram(pCPWhite);
    pRenderContext->setComputeState(pCSWhite);
    pRenderContext->setComputeVars(pCVWhite);

    //
    pRenderContext->dispatch(1, 1, 1);
    vec4 outputColorWhite = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Output Color is White.
    if (outputColorWhite != vec4(1.0, 1.0, 1.0, 1.0))
    {
        return test_fail("Output Color Not White!");
    }

    //
    //  Create the Compute Program to Grey.
    Program::DefineList defsC;
    defsC.add("_USE_COLOR_VARIABLE", "");
    defsC.add("_COLOR_VAR", "0.5");
    
    //  Create the Compute Program.
    ComputeProgram::SharedPtr pCPGrey = ComputeProgram::createFromFile("CSBase.cs.hlsl", defsC);
    ComputeVars::SharedPtr pCVGrey = ComputeVars::create(pCPGrey->getActiveVersion()->getReflector());
    ComputeState::SharedPtr pCSGrey = ComputeState::create();

    //  
    pCVGrey->setTexture("gOutput", pTexture);
    pCSGrey->setProgram(pCPGrey);
    pRenderContext->setComputeState(pCSGrey);
    pRenderContext->setComputeVars(pCVGrey);

    //
    pRenderContext->dispatch(1, 1, 1);
    vec4 outputColorGrey = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Output Color is White.
    if (outputColorGrey != vec4(0.5, 0.5, 0.5, 0.5))
    {
        return test_fail("Output Color Not Grey!");
    }

    //  Return success.
    return test_pass();
}

//  
testing_func(ProgramTest, TestComputeProgramAddRemoveDefines)
{
    //  Get the Render Context, and create the Compute Texture.
    RenderContext::SharedPtr pRenderContext = gpDevice->getRenderContext();
    Texture::SharedPtr pTexture = TestHelper::createRGBA32FRWTexture(1, 1);
    vec4 outputColor = vec4(0.0);

    //  Create the Compute Program, and create the Compute State.
    ComputeProgram::SharedPtr pCPBase = ComputeProgram::createFromFile("CSBase.cs.hlsl");
    ComputeVars::SharedPtr pCVBase = ComputeVars::create(pCPBase->getActiveVersion()->getReflector());
    ComputeState::SharedPtr pCSBase = ComputeState::create();


    //  Set the Compute State Program, and set the Compute State and Variables for Compute Program
    pCVBase->setTexture("gOutput", pTexture);
    pCSBase->setProgram(pCPBase);
    pRenderContext->setComputeState(pCSBase);
    pRenderContext->setComputeVars(pCVBase);

    //  Dispatch!
    pRenderContext->dispatch(1, 1, 1);
    outputColor = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Output Color is Black.
    if (outputColor != vec4(0.0, 0.0, 0.0, 0.0))
    {
        return test_fail("Output Color Not Black!");
    }

    //  Set to White.
    pCPBase->addDefine("_USE_WHITE");

    pRenderContext->dispatch(1, 1, 1);
    outputColor = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Output Color is Black.
    if (outputColor != vec4(1.0, 1.0, 1.0, 1.0))
    {
        return test_fail("Output Color Not White!");
    }

    //  Adding these defines will make no difference - because "_USE_WHITE" is still active.
    pCPBase->addDefine("_USE_COLOR_VARIABLE");
    pCPBase->addDefine("_COLOR_VAR", "0.5");

    pRenderContext->dispatch(1, 1, 1);
    outputColor = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Output Color is White.
    if (outputColor != vec4(1.0, 1.0, 1.0, 1.0))
    {
        return test_fail("Output Color Not White!");
    }
    
    //  
    pCPBase->clearDefines();

    //  
    for (uint32_t i = 0; i < 9; i++)
    {
        //  Add the Defines.
        pCPBase->addDefine("_USE_COLOR_VARIABLE");
        pCPBase->addDefine("_COLOR_VAR", "0." + std::to_string(i));

        //  Compute the Expected Color.
        vec4 expectedColor = vec4(((float)i) / 10.0f);

        pRenderContext->dispatch(1, 1, 1);
        outputColor = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

        //  Check if the Output Color is White.
        if (outputColor != expectedColor)
        {
            return test_fail("Output Color Incorrect!");
        }

        pCPBase->clearDefines();
    }

    //  Dispatch!
    pRenderContext->dispatch(1, 1, 1);
    outputColor = *(vec4*)((pRenderContext->readTextureSubresource(pTexture.get(), 0)).data());

    //  Check if the Output Color is Black.
    if (outputColor != vec4(0.0, 0.0, 0.0, 0.0))
    {
        return test_fail("Output Color Not Black!");
    }
    
    //  
    return test_pass();
}


//  Test the Compute Program Reload Programs.
testing_func(ProgramTest, TestComputeProgramReloadPrograms)
{
    return test_pass();
}





//  Test the Graphics Program Class.
testing_func(ProgramTest, TestGraphicsProgramCreate)
{
    return test_pass();
}

//  Test the Graphics Program Add Remove Defines.
testing_func(ProgramTest, TestGraphicsProgramAddRemoveDefines)
{
    return test_pass();
}

//  Test the Graphics Program Reload Programs.
testing_func(ProgramTest, TestGraphicsProgramReloadPrograms)
{
    return test_pass();
}




int main()
{
    ProgramTest pT;
    pT.init(true);
    pT.run();
    return 0;
}
