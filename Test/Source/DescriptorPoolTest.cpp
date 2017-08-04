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
#include "DescriptorPoolTest.h"


//  Add the Tests.
void DescriptorPoolTest::addTests()
{
    addTestToList<TestCreates>();
    addTestToList<TestDescriptorBasicReleases>();
    addTestToList<TestDescriptorCountSize>();
}

//  Test Creates
testing_func(DescriptorPoolTest, TestCreates)
{
    //
    return test_pass();

}

//  Test Simple Descriptor Set Creates.
testing_func(DescriptorPoolTest, TestDescriptorBasicReleases)
{

    //
    return test_pass();
}
//  Make sure it works for 64s.
testing_func(DescriptorPoolTest, TestDescriptorCountSize)
{

    //
    return test_pass();
}


//  Create a Descriptor Pool Desc.  
DescriptorPool::Desc DescriptorPoolTest::createDescriptorPoolDesc(std::vector<uint32_t> typeCounts)
{
    //  
    DescriptorPool::Desc dpDesc;

    //  
    for (uint32_t currentTypeIndex = 0; currentTypeIndex < DescriptorPool::kTypeCount; currentTypeIndex++)
    {
        if (typeCounts[currentTypeIndex] != 0)
        {

            dpDesc.setDescCount(DescriptorPool::Type(currentTypeIndex), typeCounts[currentTypeIndex]);
        }
    }

    //  
    return dpDesc;
}


//  Create a Descriptor Set Layout.
DescriptorSet::Layout DescriptorPoolTest::createDescriptorSetLayout(std::vector<uint32_t> typeCounts)
{
    //  Create the Descriptor Set Layouts.
    DescriptorSet::Layout dsLayout;

    //  Fill the Descriptor Set Layout.
    for (uint32_t typeIndex = 0; typeIndex < DescriptorPool::kTypeCount; typeIndex++)
    {
        dsLayout.addRange(DescriptorPool::Type(typeIndex), 0, typeCounts[typeIndex], 0);
    }

    //  
    return dsLayout;
}


int main()
{
    DescriptorPoolTest dsT;
    dsT.init(true);
    dsT.run();
    return 0;
}