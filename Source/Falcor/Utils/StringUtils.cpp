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
#include "StringUtils.h"
#include "Core/Error.h"

#include <array>
#include <string>
#include <utility>

namespace Falcor
{
bool hasPrefix(const std::string& str, const std::string& prefix, bool caseSensitive)
{
    if (str.size() >= prefix.size())
    {
        if (caseSensitive == false)
        {
            std::string s = str;
            std::string pfx = prefix;
            std::transform(str.begin(), str.end(), s.begin(), ::tolower);
            std::transform(prefix.begin(), prefix.end(), pfx.begin(), ::tolower);
            return s.compare(0, pfx.length(), pfx) == 0;
        }
        else
        {
            return str.compare(0, prefix.length(), prefix) == 0;
        }
    }
    return false;
}

bool hasSuffix(const std::string& str, const std::string& suffix, bool caseSensitive)
{
    if (str.size() >= suffix.size())
    {
        std::string s = str.substr(str.length() - suffix.length());
        if (caseSensitive == false)
        {
            std::string sfx = suffix;
            std::transform(s.begin(), s.end(), s.begin(), ::tolower);
            std::transform(sfx.begin(), sfx.end(), sfx.begin(), ::tolower);
            return (sfx == s);
        }
        else
        {
            return (s == suffix);
        }
    }
    return false;
}

std::vector<std::string> splitString(const std::string& str, const std::string& delim)
{
    std::string s;
    std::vector<std::string> vec;
    for (char c : str)
    {
        if (delim.find(c) != std::string::npos)
        {
            if (s.length())
            {
                vec.push_back(s);
                s.clear();
            }
        }
        else
        {
            s += c;
        }
    }
    if (s.length())
    {
        vec.push_back(s);
    }
    return vec;
}

std::string joinStrings(const std::vector<std::string>& strings, const std::string& separator)
{
    std::string result;
    for (auto it = strings.begin(); it != strings.end(); it++)
    {
        result += *it;

        if (it != strings.end() - 1)
        {
            result += separator;
        }
    }
    return result;
}

std::string removeLeadingWhitespace(const std::string& str, const char* whitespace)
{
    std::string result(str);
    result.erase(0, result.find_first_not_of(whitespace));
    return result;
}

std::string removeTrailingWhitespace(const std::string& str, const char* whitespace)
{
    std::string result(str);
    result.erase(result.find_last_not_of(whitespace) + 1);
    return result;
}

std::string removeLeadingTrailingWhitespace(const std::string& str, const char* whitespace)
{
    return removeTrailingWhitespace(removeLeadingWhitespace(str, whitespace), whitespace);
}

std::string replaceCharacters(const std::string& str, const char* characters, const char replacement)
{
    std::string result(str);
    size_t pos = result.find_first_of(characters);
    while (pos != std::string::npos)
    {
        result[pos] = replacement;
        pos = result.find_first_of(characters, pos);
    }
    return result;
}

std::string padStringToLength(const std::string& str, size_t length, char padding)
{
    std::string result = str;
    if (result.length() < length)
        result.resize(length, padding);
    return result;
}

std::string replaceSubstring(const std::string& input, const std::string& src, const std::string& dst)
{
    std::string res = input;
    size_t offset = res.find(src);
    while (offset != std::string::npos)
    {
        res.replace(offset, src.length(), dst);
        offset += dst.length();
        offset = res.find(src, offset);
    }
    return res;
}

std::string decodeURI(const std::string& input)
{
    std::string result;
    for (size_t i = 0; i < input.length(); i++)
    {
        if (input[i] == '%')
        {
            if (i + 2 < input.length())
            {
                std::string hex = input.substr(i + 1, 2);
                char c = static_cast<char>(strtol(hex.c_str(), nullptr, 16));
                result += c;
                i += 2;
            }
        }
        else if (input[i] == '+')
        {
            result += ' ';
        }
        else
        {
            result += input[i];
        }
    }
    return result;
}

bool parseArrayIndex(const std::string& name, std::string& nonArray, uint32_t& index)
{
    size_t dot = name.find_last_of('.');
    size_t bracket = name.find_last_of('[');

    if (bracket != std::string::npos)
    {
        // Ignore cases where the last index is an array of struct index (SomeStruct[1].v should be ignored)
        if ((dot == std::string::npos) || (bracket > dot))
        {
            // We know we have an array index. Make sure it's in range
            std::string indexStr = name.substr(bracket + 1);
            char* pEndPtr;
            index = strtol(indexStr.c_str(), &pEndPtr, 0);
            FALCOR_ASSERT(*pEndPtr == ']');
            nonArray = name.substr(0, bracket);
            return true;
        }
    }

    return false;
}

void copyStringToBuffer(char* buffer, uint32_t bufferSize, const std::string& s)
{
    const uint32_t length = std::min(bufferSize - 1, (uint32_t)s.length());
    s.copy(buffer, length);
    buffer[length] = '\0';
}

std::string formatByteSize(size_t size)
{
    if (size < 1024ull)
        return fmt::format("{} B", size);
    else if (size < 1048576ull)
        return fmt::format("{:.2f} kB", size / 1024.0);
    else if (size < 1073741824ull)
        return fmt::format("{:.2f} MB", size / 1048576.0);
    else if (size < 1099511627776ull)
        return fmt::format("{:.2f} GB", size / 1073741824.0);
    else
        return fmt::format("{:.2f} TB", size / 1099511627776.0);
}

std::string encodeBase64(const void* data, size_t len)
{
    // based on https://gist.github.com/tomykaira/f0fd86b6c73063283afe550bc5d77594
    // clang-format off
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
    // clang-format on

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
    // clang-format off
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
    // clang-format on

    size_t inLen = in.size();
    if (inLen == 0)
        return {};
    if (inLen % 4 != 0)
        FALCOR_THROW("Input data size is not a multiple of 4");

    size_t outLen = inLen / 4 * 3;
    if (in[inLen - 1] == '=')
        outLen--;
    if (in[inLen - 2] == '=')
        outLen--;

    std::vector<uint8_t> out(outLen, 0);

    for (size_t i = 0, j = 0; i < inLen;)
    {
        uint32_t a = in[i] == '=' ? 0 & i++ : kDecodingTable[static_cast<uint32_t>(in[i++])];
        uint32_t b = in[i] == '=' ? 0 & i++ : kDecodingTable[static_cast<uint32_t>(in[i++])];
        uint32_t c = in[i] == '=' ? 0 & i++ : kDecodingTable[static_cast<uint32_t>(in[i++])];
        uint32_t d = in[i] == '=' ? 0 & i++ : kDecodingTable[static_cast<uint32_t>(in[i++])];

        uint32_t triple = (a << 3 * 6) + (b << 2 * 6) + (c << 1 * 6) + (d << 0 * 6);

        if (j < outLen)
            out[j++] = (triple >> 2 * 8) & 0xff;
        if (j < outLen)
            out[j++] = (triple >> 1 * 8) & 0xff;
        if (j < outLen)
            out[j++] = (triple >> 0 * 8) & 0xff;
    }

    return out;
}
} // namespace Falcor
