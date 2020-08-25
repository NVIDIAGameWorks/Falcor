/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/AlignedAllocator.h"

namespace Falcor
{
    template <int N> struct SizedStruct
    {
        char buf[N];
    };

    CPU_TEST(AlignedAllocator)
    {
        AlignedAllocator alloc;
        alloc.setMinimumAlignment(16);
        alloc.setCacheLineSize(128);
        alloc.reserve(1024);
        EXPECT_EQ(1024, alloc.getCapacity());
        EXPECT_EQ(0, alloc.getSize());

        // Do an initial 15 byte allocation. Make sure that everything
        // makes sense.
        EXPECT_EQ(15, sizeof(SizedStruct<15>));
        void* ptr = alloc.allocate<SizedStruct<15>>();
        EXPECT_EQ(15, alloc.getSize());
        EXPECT_EQ(0, alloc.offsetOf(ptr));
        EXPECT_EQ(0, reinterpret_cast<char*>(ptr) - reinterpret_cast<char*>(alloc.getStartPointer()));

        // Allocate another 8 bytes. Make sure it starts 16-byte aligned.
        ptr = alloc.allocate<SizedStruct<8>>();
        EXPECT_EQ(24, alloc.getSize());
        EXPECT_EQ(16, alloc.offsetOf(ptr));

        // Do a one byte allocation and make sure it also starts aligned.
        ptr = alloc.allocate<char>();
        EXPECT_EQ(33, alloc.getSize());
        EXPECT_EQ(32, alloc.offsetOf(ptr));

        // A 100 byte allocation should start at a new cache line now.
        ptr = alloc.allocate<SizedStruct<100>>();
        EXPECT_EQ(128, alloc.offsetOf(ptr));
        EXPECT_EQ(228, alloc.getSize());
    }

    CPU_TEST(AlignedAllocatorNoCacheLine)
    {
        AlignedAllocator alloc;
        alloc.setMinimumAlignment(16);
        alloc.setCacheLineSize(0);      // Don't worry about allocations that span cache lines.
        alloc.reserve(1024);
        EXPECT_EQ(1024, alloc.getCapacity());
        EXPECT_EQ(0, alloc.getSize());

        void* ptr = alloc.allocate<SizedStruct<64>>();
        EXPECT_EQ(64, alloc.getSize());
        EXPECT_EQ(0, alloc.offsetOf(ptr));

        // Now allocate 72 bytes. It should be immediately after the
        // initial allocation since we're already aligned.
        ptr = alloc.allocate<SizedStruct<72>>();
        EXPECT_EQ(64+72, alloc.getSize());
        EXPECT_EQ(64, alloc.offsetOf(ptr));
    }
}
