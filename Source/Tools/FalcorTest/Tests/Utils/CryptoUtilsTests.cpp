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
#include "Utils/CryptoUtils.h"
#include <random>

namespace Falcor
{
    CPU_TEST(SHA1)
    {
        {
            std::string str1{"Hello "};
            std::string str2{"World!"};
            SHA1::MD md{0x2e, 0xf7, 0xbd, 0xe6, 0x08, 0xce, 0x54, 0x04, 0xe9, 0x7d, 0x5f, 0x04, 0x2f, 0x95, 0xf8, 0x9f, 0x1c, 0x23, 0x28, 0x71};
            SHA1 sha1;
            sha1.update(str1.data(), str1.size());
            sha1.update(str2.data(), str2.size());
            EXPECT(sha1.finalize() == md);
        }

        {
            std::string str{"Hello World!"};
            SHA1::MD md{0x2e, 0xf7, 0xbd, 0xe6, 0x08, 0xce, 0x54, 0x04, 0xe9, 0x7d, 0x5f, 0x04, 0x2f, 0x95, 0xf8, 0x9f, 0x1c, 0x23, 0x28, 0x71};
            EXPECT(SHA1::compute(str.data(), str.size()) == md);
        }

        {
            std::string str{"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."};
            SHA1::MD md{0xcd, 0x36, 0xb3, 0x70, 0x75, 0x8a, 0x25, 0x9b, 0x34, 0x84, 0x50, 0x84, 0xa6, 0xcc, 0x38, 0x47, 0x3c, 0xb9, 0x5e, 0x27};
            EXPECT(SHA1::compute(str.data(), str.size()) == md);
        }
    }
}
