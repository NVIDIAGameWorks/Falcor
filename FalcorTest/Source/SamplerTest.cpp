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
#include "SamplerTest.h"

void SamplerTest::addTests()
{
    addTestToList<TestCreate>();
}

testing_func(SamplerTest, TestCreate)
{
    const uint32_t numFilterCombinations = 8;
    const Sampler::Filter point = Sampler::Filter::Point;
    const Sampler::Filter linear = Sampler::Filter::Linear;
    const Sampler::Filter minFilters[numFilterCombinations] = { point, linear, point, point, linear, point, linear, linear };
    const Sampler::Filter magFilters[numFilterCombinations] = { point, linear, point, linear, point, linear, point, linear };
    const Sampler::Filter mipFilters[numFilterCombinations] = { point, linear, linear, point, point, linear, linear, point };
    const uint32_t numComparionModes = 9;
    const uint32_t numAddressModes = 5;
    const uint32_t numRandomNumbers = 10;
    
    TestDesc desc;
    //Tests nearly all combinations of settings
    //Filter
    for (uint32_t i = 0; i < numFilterCombinations; ++i)
    {
        desc.setFilterMode(minFilters[i], magFilters[i], mipFilters[i]);
        //comparison mode
        for (uint32_t j = 0; j < numComparionModes; ++j)
        {
            desc.setComparisonMode(static_cast<Sampler::ComparisonMode>(j));
            //random numbers for border color, minLod, maxLod, lodBias, and max anisotropy
            for (uint32_t k = 0; k < numRandomNumbers; ++k)
            {
                float colorR = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                float colorG = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                float colorB = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                float colorA = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                desc.setBorderColor(glm::vec4(colorR, colorG, colorB, colorA));

                float minLod = static_cast<float>(rand());
                float maxLod = static_cast<float>(rand());
                float lodBias = static_cast<float>(rand());
                desc.setLodParams(minLod, maxLod, lodBias);

                uint32_t maxAnisotropy = static_cast<uint32_t>(rand());
                desc.setMaxAnisotropy(maxAnisotropy);

                //mode u 
                for (uint32_t x = 0; x < numAddressModes; ++x)
                {
                    //mode v
                    for (uint32_t y = 0; y < numAddressModes; ++y)
                    {
                        //mode w
                        for (uint32_t z = 0; z < numAddressModes; ++z)
                        {
                            desc.setAddressingMode(
                                static_cast<Sampler::AddressMode>(x), 
                                static_cast<Sampler::AddressMode>(y), 
                                static_cast<Sampler::AddressMode>(z));

                            //create sampler and ensure matches desc used to create
                            Sampler::SharedPtr sampler = Sampler::create(desc);
                            if (!doStatesMatch(sampler, desc))
                            {
                                return test_fail("Sampler doesn't match desc used to create it");
                            }
                        }
                    }
                }
            }
        }
    }

    return test_pass();
}

bool SamplerTest::doStatesMatch(Sampler::SharedPtr sampler, TestDesc desc)
{
    return sampler->getMagFilter() == desc.mMagFilter &&
        sampler->getMinFilter() == desc.mMinFilter &&
        sampler->getMipFilter() == desc.mMipFilter &&
        sampler->getMaxAnisotropy() == desc.mMaxAnisotropy &&
        sampler->getMinLod() == desc.mMinLod &&
        sampler->getMaxLod() == desc.mMaxLod &&
        sampler->getLodBias() == desc.mLodBias &&
        sampler->getComparisonMode() == desc.mComparisonMode &&
        sampler->getAddressModeU() == desc.mModeU &&
        sampler->getAddressModeV() == desc.mModeV &&
        sampler->getAddressModeW() == desc.mModeW &&
        sampler->getBorderColor() == desc.mBorderColor;
}

int main()
{
    SamplerTest st;
    st.init(true);
    st.run();
    return 0;
}
