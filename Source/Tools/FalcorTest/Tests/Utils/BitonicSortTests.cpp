/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "Testing/UnitTest.h"
#include "Utils/Algorithm/BitonicSort.h"
#include <random>

namespace Falcor
{
    namespace
    {
        // Sort the 'data' array in ascending order within chunks of 'chunkSize' elements.
        void sort(std::vector<uint32_t>& data, const uint32_t chunkSize)
        {
            if (chunkSize <= 1) return;
            for (size_t first = 0; first < data.size(); first += chunkSize)
            {
                size_t last = std::min(first + chunkSize, data.size());
                std::sort(data.begin() + first, data.begin() + last);
            }
        }

        void testGpuSort(GPUUnitTestContext& ctx, BitonicSort* pSort, const uint32_t n, const uint32_t chunkSize)
        {
            // Create a buffer of random data to use as test data.
            std::vector<uint32_t> testData(n);
            std::mt19937 r;
            for (auto& it : testData) it = r();

            Buffer::SharedPtr pTestDataBuffer = Buffer::create(n * sizeof(uint32_t), Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None, testData.data());

            // Execute sort on the GPU.
            uint32_t groupSize = std::max(chunkSize, 256u);
            bool retval = pSort->execute(ctx.getRenderContext(), pTestDataBuffer, n, chunkSize, groupSize);
            EXPECT_EQ(retval, true);

            // Sort the test data on the CPU for comparison.
            sort(testData, chunkSize);

            // Compare results.
            const uint32_t* result = (const uint32_t*)pTestDataBuffer->map(Buffer::MapType::Read);
            FALCOR_ASSERT(result);
            for (uint32_t i = 0; i < n; i++)
            {
                EXPECT_EQ(testData[i], result[i]) << "i = " << i;
            }
            pTestDataBuffer->unmap();
        }
    }

#if FALCOR_NVAPI_AVAILABLE
    GPU_TEST(BitonicSort)
#else
    GPU_TEST(BitonicSort, "Requires NVAPI")
#endif
    {
        // Create utility class for sorting.
        BitonicSort::SharedPtr pSort = BitonicSort::create();

        // Test different parameters.
        // The chunk size(last param) must  be a pow-of-two <= 1024.
        testGpuSort(ctx, pSort.get(), 100, 1);
        testGpuSort(ctx, pSort.get(), 19, 2);
        testGpuSort(ctx, pSort.get(), 1024, 4);
        testGpuSort(ctx, pSort.get(), 11025, 8);
        testGpuSort(ctx, pSort.get(), 290, 16);
        testGpuSort(ctx, pSort.get(), 1500, 32);
        testGpuSort(ctx, pSort.get(), 20000, 64);
        testGpuSort(ctx, pSort.get(), 2001, 128);
        testGpuSort(ctx, pSort.get(), 16384, 256);
        testGpuSort(ctx, pSort.get(), 3103, 1024);
    }
}
