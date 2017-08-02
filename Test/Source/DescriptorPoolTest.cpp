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
    srand(100);

    //  Types Max Count Corresponding to the Enum Order.
    std::vector<uint32_t> shaderVisibleTypesMaxCount = { 10, 10, 10, 0, 0, 10 };
    std::vector<uint32_t> shaderNonVisibleTypesMaxCount = { 10, 10, 10, 1, 0, 10 };

    //  
    uint32_t maxTypeCount = 10;
    

    //  Shader Visible Types
    for (uint32_t i = 0; i < DescriptorPool::kTypeCount; i++)
    {   
        //  
        bool isUAV = false;
        isUAV = isUAV && ((DescriptorPool::Type)i) == DescriptorPool::Type::TypedBufferUav;
        isUAV = isUAV && ((DescriptorPool::Type)i) == DescriptorPool::Type::StructuredBufferUav;
        isUAV = isUAV && ((DescriptorPool::Type)i) == DescriptorPool::Type::TextureUav;

        //  
        if (isUAV)
        {
            shaderVisibleTypesMaxCount.push_back(maxTypeCount);
        }
        else
        {

        }
    }

    //  Construct the GpuFence.
    GpuFence::SharedPtr gpuFence = GpuFence::create();

    //  
    for (uint32_t sVCount = 0; sVCount < 2; sVCount++)
    {
        //  Compute the number of all possible type combinations.
        uint32_t typeCombinations = (uint32_t)std::pow(2, DescriptorPool::kTypeCount);

        //  
        for (uint32_t currentTypeSet = 0; currentTypeSet < typeCombinations; currentTypeSet++)
        {
            //  Create the Descriptor Pool Descriptor.
            DescriptorPool::Desc dpDesc;
            dpDesc.setShaderVisible(sVCount == 1);

            //  
            std::vector<uint32_t> typesMaxCount(DescriptorPool::kTypeCount);

            if (sVCount == 1)
            {
                typesMaxCount = shaderVisibleTypesMaxCount;
            }
            else
            {
                typesMaxCount = shaderNonVisibleTypesMaxCount;
            }

            //  
            for (uint32_t currentTypeIndex = 0; currentTypeIndex < DescriptorPool::kTypeCount; currentTypeIndex++)
            {
                //  
                if ((currentTypeSet & (uint32_t)std::pow(2, currentTypeIndex)) == (uint32_t)std::pow(2, currentTypeIndex))
                {
                    uint32_t count = 0;

                    //  
                    if (typesMaxCount[currentTypeIndex] != 0)
                    {
                        count = rand() % (typesMaxCount[currentTypeIndex] + 1);
                        //  
                        if (count != 0)
                        {
                            dpDesc.setDescCount(DescriptorPool::Type(currentTypeIndex), count);
                        }
                        else
                        {
                            dpDesc.setDescCount(DescriptorPool::Type(currentTypeIndex), 1);
                        }
                    }
                }
            }

            //  Create the Descriptor Pool.
            DescriptorPool::SharedPtr pDP = DescriptorPool::create(dpDesc, gpuFence);

            //  Check if the Descriptor Pool Created.
            if (!pDP)
            {
                //  Return failure.    
                return test_fail("Failed To Create Descriptor Pool!")
            }
            else
            {
                //  Verify.
                bool verifySuccess = true;
                verifySuccess = verifySuccess && (pDP->getApiData() != nullptr);

                //  
                if (!verifySuccess)
                {
                    return test_fail("Created Descriptor Pool, but API Data is null");
                }
            }
        }
    }

    //  
    return test_pass();
}


//  Test Simple Descriptor Set Creates.
testing_func(DescriptorPoolTest, TestDescriptorBasicReleases)
{
    srand(100);

    // Create a number of steps.
    uint32_t steps = 20;
    uint32_t typeStep = 128;
    uint32_t countPerType = steps * typeStep;

    //  Construct the GpuFence.
    GpuFence::SharedPtr gpuFence = GpuFence::create();


    std::vector<uint32_t> countsPerType;
    
    for (uint32_t i = 0; i < DescriptorPool::kTypeCount; i++)
    {
        countsPerType.push_back(countPerType);
    }
    


    //
    DescriptorPool::Desc dpDesc = createDescriptorPoolDesc(countsPerType);
    dpDesc.setShaderVisible(false);

    //  Create the Descriptor Pool.
    DescriptorPool::SharedPtr pDP = DescriptorPool::create(dpDesc, gpuFence);

    //  Create the Descriptor Set Layout.
    DescriptorSet::Layout dslayout = createDescriptorSetLayout({ typeStep , typeStep , typeStep , typeStep , typeStep , typeStep });

    //  
    for (uint32_t stepIndex = 0; stepIndex < steps; stepIndex++)
    {
        //  Create the Descriptor Set.
        DescriptorSet::SharedPtr pDS = DescriptorSet::create(pDP, dslayout);

        //  
        pDS = nullptr;

        //  
        if (pDP->getDeferredReleasesSize() != (stepIndex + 1))
        {
            return test_fail("Deferred Releases Queue size incorrect!");
        }
    }

    //  Execute the Deferred Releases.
    pDP->executeDeferredReleases();

    //  Confirm that all the Deferred Resources have been released.
    if (pDP->getDeferredReleasesSize() != 0)
    {
        return test_fail("Deferred Releases Incomplete!");
    }

    //
    return test_pass();
}
//  Make sure it works for 64s.
testing_func(DescriptorPoolTest, TestDescriptorCountSize)
{

    for (uint32_t allocPow = 0; allocPow < 4; allocPow++)
    {
        //  
        uint32_t allocsize = (uint32_t)std::pow(2, (allocPow + 1));
        uint32_t dsPerBatchCount = 64 / allocsize;
        uint32_t dsBatches = 2 * 10;

        uint32_t poolDescCount = allocsize * dsPerBatchCount * (dsBatches - 1);


        //  Construct the GpuFence.
        GpuFence::SharedPtr gpuFence = GpuFence::create();

        //  Create the Descriptor Pool Descriptor.
        DescriptorPool::Desc dpDesc = createDescriptorPoolDesc({ poolDescCount, poolDescCount, poolDescCount, poolDescCount, poolDescCount, poolDescCount });
        dpDesc.setShaderVisible(false);
        DescriptorPool::SharedPtr pDP = DescriptorPool::create(dpDesc, gpuFence);

        //  Create the Descriptor Set Layout.
        DescriptorSet::Layout dsLayout = createDescriptorSetLayout({ allocsize , allocsize , allocsize , allocsize , allocsize , allocsize });
        std::vector<DescriptorSet::SharedPtr> dsSets(dsPerBatchCount * dsBatches);

        //  Create the Descriptor Set Batches, execpt for one.
        for (uint32_t currentBatchIndex = 0; currentBatchIndex < dsBatches - 1; currentBatchIndex++)
        {
            //  
            for (uint32_t currentDSIndex = 0; currentDSIndex < dsPerBatchCount; currentDSIndex++)
            {
                DescriptorSet::SharedPtr dsSet = DescriptorSet::create(pDP, dsLayout);
                dsSets[currentBatchIndex * dsPerBatchCount + currentDSIndex] = dsSet;
            }
            //  
        }


        uint32_t nextActive = dsBatches - 1;
        uint32_t nextInactive = dsBatches - 2;
        uint32_t testsCount = 50;

        //  
        for (uint32_t currentIndex = 0; currentIndex < testsCount; currentIndex++)
        {
            //  
            for (uint32_t currentDSIndex = 0; currentDSIndex < dsPerBatchCount; currentDSIndex++)
            {
                dsSets[nextInactive * dsPerBatchCount + currentDSIndex] = nullptr;
            }

            //  
            for (uint32_t currentDSIndex = 0; currentDSIndex < dsPerBatchCount; currentDSIndex++)
            {
                DescriptorSet::SharedPtr dsSet = DescriptorSet::create(pDP, dsLayout);
                dsSets[nextActive * dsPerBatchCount + currentDSIndex] = dsSet;
            }

            //  Reset the Index.
            if (nextActive == 0)
            {
                nextActive = dsBatches - 1;
            }
            else
            {
                nextActive = nextActive - 1;
            }

            if (nextInactive == 0)
            {
                nextInactive = dsBatches - 1;
            }
            else
            {
                nextInactive = nextInactive - 1;
            }
        }

        //  
        for (uint32_t currentBatchIndex = 0; currentBatchIndex < dsBatches - 1; currentBatchIndex++)
        {
            //  
            for (uint32_t currentDSIndex = 0; currentDSIndex < dsPerBatchCount; currentDSIndex++)
            {
                dsSets[currentBatchIndex * dsPerBatchCount + currentDSIndex] = nullptr;
            }
        }

        //  Execute the Deferred Releases.
        pDP->executeDeferredReleases();

        //  Confirm that all the Resources have been released.
        if (pDP->getDeferredReleasesSize() != 0)
        {
            return test_fail("Deferred Releases Incomplete!");
        }
    }
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