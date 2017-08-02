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
#include "ComputeStateObjectTest.h"

void ComputeStateObjectTest::addTests()
{
    addTestToList<TestAll>();

}


ComputeStateObject::SharedPtr ComputeStateObjectTest::createCSO(const std::string & sourceFile, ComputeStateObject::Desc & csDesc)
{
    //  Create the Compute State Object for the Shader.
    ComputeProgram::SharedPtr pCP = ComputeProgram::createFromFile(sourceFile);
    RootSignature::SharedPtr pRS = RootSignature::create(pCP->getActiveVersion()->getReflector().get());

    //  Create the Compute State Object Desc.
    csDesc.setProgramVersion(pCP->getActiveVersion());
    csDesc.setRootSignature(pRS);

    //  Create the Compute State Object.
    ComputeStateObject::SharedPtr pCS = ComputeStateObject::create(csDesc);

    //  
    if (pCS == nullptr)
    {
        return nullptr;
    }

    //
    if (!(csDesc == pCS->getDesc()))
    {
        return nullptr;
    }

    return pCS;
}

testing_func(ComputeStateObjectTest, TestAll)
{
    ComputeStateObject::Desc pCSDB;
    ComputeStateObject::SharedPtr pCSBlack = createCSO("CSBlack.cs.hlsl", pCSDB);
    
    ComputeStateObject::Desc pCSDW;
    ComputeStateObject::SharedPtr pCSWhite = createCSO("CSWhite.cs.hlsl", pCSDW);

    //  
    if (pCSBlack == nullptr) return test_fail("Failed to create Compute State Object - CSBlack.cs.hlsl!");
    if (pCSWhite == nullptr) return test_fail("Failed to create Compute State Object - CSWhite.cs.hlsl!");

    //
    if (!(pCSBlack->getDesc() == pCSDB)) return test_fail("Created Desc does not match provided Desc  CSBlack.cs.hlsl!");
    if (!(pCSWhite->getDesc() == pCSDW)) return test_fail("Created Desc does not match provided Desc  CSWhite.cs.hlsl!");


    return test_pass();
}

int main()
{
    ComputeStateObjectTest csoT;
    csoT.init(true);
    csoT.run();
    return 0;
}
