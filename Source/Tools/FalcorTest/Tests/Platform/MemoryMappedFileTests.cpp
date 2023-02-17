/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/Platform/OS.h"
#include "Core/Platform/MemoryMappedFile.h"

#include <vector>
#include <fstream>
#include <random>

namespace Falcor
{
CPU_TEST(MemoryMappedFile_Closed)
{
    MemoryMappedFile file;
    EXPECT_EQ(file.isOpen(), false);
    EXPECT_EQ(file.getSize(), 0);
    EXPECT_EQ(file.getData(), nullptr);
    EXPECT_EQ(file.getMappedSize(), 0);

    // Allowed to close a closed file.
    file.close();
    EXPECT_EQ(file.isOpen(), false);
}

CPU_TEST(MemoryMappedFile_NonExisting)
{
    {
        MemoryMappedFile file;
        EXPECT_EQ(file.open("__file_that_does_not_exist__"), false);
        EXPECT_EQ(file.isOpen(), false);
    }

    {
        MemoryMappedFile file("__file_that_does_not_exist__");
        EXPECT_EQ(file.isOpen(), false);
    }
}

CPU_TEST(MemoryMappedFile_Read)
{
    std::vector<uint8_t> randomData(128 * 1024);
    std::mt19937 rng;
    for (size_t i = 0; i < randomData.size(); ++i)
        randomData[i] = rng() & 0xff;

    const std::filesystem::path tempPath = std::filesystem::absolute("test_memory_mapped.bin");

    // Write file with random data.
    std::ofstream ofs(tempPath, std::ios::binary);
    ASSERT_TRUE(ofs.good());
    ofs.write(reinterpret_cast<const char*>(randomData.data()), randomData.size());
    ofs.close();

    {
        // Map entire file.
        MemoryMappedFile file(tempPath);
        EXPECT_EQ(file.isOpen(), true);
        EXPECT_EQ(file.getSize(), randomData.size());
        EXPECT_NE(file.getData(), nullptr);
        EXPECT_GE(file.getMappedSize(), randomData.size());
        EXPECT(std::memcmp(file.getData(), randomData.data(), file.getSize()) == 0);
    }

    {
        // Map first 1024 bytes.
        MemoryMappedFile file(tempPath, 1024);
        EXPECT_EQ(file.isOpen(), true);
        EXPECT_EQ(file.getSize(), randomData.size());
        EXPECT_NE(file.getData(), nullptr);
        EXPECT_GE(file.getMappedSize(), 1024);
        EXPECT(std::memcmp(file.getData(), randomData.data(), 1024) == 0);
    }

    {
        // Map first page.
        size_t pageSize = MemoryMappedFile::getPageSize();
        EXPECT_GE(pageSize, 4096);
        ASSERT_LE(pageSize, randomData.size());
        MemoryMappedFile file(tempPath, pageSize);
        EXPECT_EQ(file.isOpen(), true);
        EXPECT_EQ(file.getSize(), randomData.size());
        EXPECT_NE(file.getData(), nullptr);
        EXPECT_GE(file.getMappedSize(), pageSize);
        EXPECT(std::memcmp(file.getData(), randomData.data(), pageSize) == 0);
    }

    // Cleanup.
    std::filesystem::remove(tempPath);
}

} // namespace Falcor
