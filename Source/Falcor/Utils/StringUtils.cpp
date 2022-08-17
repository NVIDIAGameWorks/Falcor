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
#include "StringUtils.h"

#include <string>
#include <utility>

namespace Falcor
{
    std::string formatByteSize(size_t size)
    {
        std::array<std::pair<size_t, std::string>, 5> memorySizes =
        {
            std::make_pair(UINT64_C(1), "B"),
            std::make_pair(UINT64_C(1024), "kB"),
            std::make_pair(UINT64_C(1048576), "MB"),
            std::make_pair(UINT64_C(1073741824), "GB"),
            std::make_pair(UINT64_C(1073741824)*1024, "TB")
        };

        // We could use some tricks to count zero bits from the left for a non-looped version,
        // but this is fast enough and obvious enough
        unsigned chosenSize = 0;
        for(; chosenSize < memorySizes.size() - 1; ++chosenSize)
        {
            if (memorySizes[chosenSize].first < size && size < memorySizes[chosenSize + 1].first)
                break;
        }

        return fmt::format("{:.3f} {}", double(size) / memorySizes[chosenSize].first, memorySizes[chosenSize].second);
    }

    std::string encodeBase64(const void* data, size_t len)
    {
        // based on https://gist.github.com/tomykaira/f0fd86b6c73063283afe550bc5d77594
        static constexpr char kEncodingTable[] = {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
            'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
            'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
            'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z', '0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9', '+', '/'
        };

        size_t outLen = 4 * ((len + 2) / 3);
        std::string out(outLen, '\0');

        const uint8_t* pIn = reinterpret_cast<const uint8_t*>(data);
        auto pOut = out.data();

        size_t i;
        for (i = 0; i + 2 < len; i += 3)
        {
            *pOut++ = kEncodingTable[(pIn[i] >> 2) & 0x3f];
            *pOut++ = kEncodingTable[((pIn[i] & 0x3) << 4) | ((pIn[i + 1] & 0xf0) >> 4)];
            *pOut++ = kEncodingTable[((pIn[i + 1] & 0xf) << 2) | ((pIn[i + 2] & 0xc0) >> 6)];
            *pOut++ = kEncodingTable[pIn[i + 2] & 0x3f];
        }
        if (i < len)
        {
            *pOut++ = kEncodingTable[(pIn[i] >> 2) & 0x3f];
            if (i == (len - 1))
            {
                *pOut++ = kEncodingTable[((pIn[i] & 0x3) << 4)];
                *pOut++ = '=';
            }
            else
            {
                *pOut++ = kEncodingTable[((pIn[i] & 0x3) << 4) | ((pIn[i + 1] & 0xf0) >> 4)];
                *pOut++ = kEncodingTable[((pIn[i + 1] & 0xf) << 2)];
            }
            *pOut++ = '=';
        }

        return out;
    }

    std::vector<uint8_t> decodeBase64(const std::string& in)
    {
        // based on https://gist.github.com/tomykaira/f0fd86b6c73063283afe550bc5d77594
        static constexpr uint8_t kDecodingTable[] = {
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 62, 64, 64, 64, 63,
            52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 64, 64, 64, 64, 64, 64,
            64,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14,
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 64, 64, 64, 64, 64,
            64, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
            64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64
        };

        size_t inLen = in.size();
        if (inLen == 0) return {};
        if (inLen % 4 != 0) throw ArgumentError("Input data size is not a multiple of 4");

        size_t outLen = inLen / 4 * 3;
        if (in[inLen - 1] == '=') outLen--;
        if (in[inLen - 2] == '=') outLen--;

        std::vector<uint8_t> out(outLen, 0);

        for (size_t i = 0, j = 0; i < inLen;)
        {
            uint32_t a = in[i] == '=' ? 0 & i++ : kDecodingTable[static_cast<uint32_t>(in[i++])];
            uint32_t b = in[i] == '=' ? 0 & i++ : kDecodingTable[static_cast<uint32_t>(in[i++])];
            uint32_t c = in[i] == '=' ? 0 & i++ : kDecodingTable[static_cast<uint32_t>(in[i++])];
            uint32_t d = in[i] == '=' ? 0 & i++ : kDecodingTable[static_cast<uint32_t>(in[i++])];

            uint32_t triple = (a << 3 * 6) + (b << 2 * 6) + (c << 1 * 6) + (d << 0 * 6);

            if (j < outLen) out[j++] = (triple >> 2 * 8) & 0xff;
            if (j < outLen) out[j++] = (triple >> 1 * 8) & 0xff;
            if (j < outLen) out[j++] = (triple >> 0 * 8) & 0xff;
        }

        return out;
    }
}
