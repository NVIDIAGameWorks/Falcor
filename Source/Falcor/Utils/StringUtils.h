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
#pragma once
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <locale>
#include <codecvt>

namespace Falcor
{
    // String/string_View append operators missing from the spec
    inline std::string operator+(const std::string& lhs, std::string_view rhs)
    {
        std::string s = lhs;
        s.append(rhs);
        return s;
    }

    inline std::string& operator+=(std::string& lhs, std::string_view rhs)
    {
        lhs.append(rhs);
        return lhs;
    }

    /*!
    *  \addtogroup Falcor
    *  @{
    */

    /** Check is a string starts with another string
        \param[in] str String to check in
        \param[in] prefix Prefix to check for
        \param[in] caseSensitive Whether comparison should be case-sensitive
        \return Returns true if string starts with the specified prefix.
    */
    inline bool hasPrefix(const std::string& str, const std::string& prefix, bool caseSensitive = true)
    {
        if(str.size() >= prefix.size())
        {
            if(caseSensitive == false)
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

    /** Check is a string ends with another string
        \param[in] str String to check in
        \param[in] suffix Suffix to check for
        \param[in] caseSensitive Whether comparison should be case-sensitive
        \return Returns true if string ends with the specified suffix
    */
    inline bool hasSuffix(const std::string& str, const std::string& suffix, bool caseSensitive = true)
    {
        if(str.size() >= suffix.size())
        {
            std::string s = str.substr(str.length() - suffix.length());
            if(caseSensitive == false)
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

    /** Split a string into a vector of strings based on d delimiter
        \param[in] str String to split
        \param[in] delim Delimiter to split strings by
        \return Array of split strings excluding delimiters.
    */
    inline std::vector<std::string> splitString(const std::string& str, const std::string& delim)
    {
        std::string s;
        std::vector<std::string> vec;
        for(char c : str)
        {
            if(delim.find(c) != std::string::npos)
            {
                if(s.length())
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
        if(s.length())
        {
            vec.push_back(s);
        }
        return vec;
    }

    /** Join an array of strings separated by another set string
        \param[in] strings Array of strings to join.
        \param[in] separator String placed between each string to be joined.
        \return Joined string.
    */
    inline std::string joinStrings(const std::vector<std::string>& strings, const std::string& separator)
    {
        std::string result;
        for(auto it = strings.begin(); it != strings.end(); it++)
        {
            result += *it;

            if(it != strings.end() - 1)
            {
                result += separator;
            }
        }
        return result;
    }

    /** Remove leading whitespace.
        \param[in] str String to operate on.
        \param[in] whitespace Whitespace characters.
        \return String with leading whitespace removed.
    */
    inline std::string removeLeadingWhitespace(const std::string& str, const char* whitespace = " \n\r\t")
    {
        std::string result(str);
        result.erase(0, result.find_first_not_of(whitespace));
        return result;
    }

    /** Remove trailing whitespace.
        \param[in] str String to operate on.
        \param[in] whitespace Whitespace characters.
        \return String with trailing whitespace removed.
    */
    inline std::string removeTrailingWhitespace(const std::string& str, const char* whitespace = " \n\r\t")
    {
        std::string result(str);
        result.erase(result.find_last_not_of(whitespace) + 1);
        return result;
    }

    /** Remove leading and trailing whitespace.
        \param[in] str String to operate on.
        \param[in] whitespace Whitespace characters.
        \return String with leading and trailing whitespace removed.
    */
    inline std::string removeLeadingTrailingWhitespace(const std::string& str, const char* whitespace = " \n\r\t")
    {
        return removeTrailingWhitespace(removeLeadingWhitespace(str, whitespace), whitespace);
    }

    /** Pad string to minimum length.
    */
    inline std::string padStringToLength(const std::string& str, size_t length, char padding = ' ')
    {
        std::string result = str;
        if (result.length() < length) result.resize(length, padding);
        return result;
    }

    /** Replace all occurrences of a substring in a string. The function doesn't change the original string.
        \param input The input string
        \param src The substring to replace
        \param dst The substring to replace Src with
    */
    inline std::string replaceSubstring(const std::string& input, const std::string& src, const std::string& dst)
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

    /** Parses a string in the format <name>[<index>]. If format is valid, outputs the base name and the array index.
        \param[in] name String to parse
        \param[out] nonArray Becomes set to the non-array index portion of the string
        \param[out] index Becomes set to the index value parsed from the string
        \return Whether string was successfully parsed.
    */
    inline bool parseArrayIndex(const std::string& name, std::string& nonArray, uint32_t& index)
    {
        size_t dot = name.find_last_of('.');
        size_t bracket = name.find_last_of('[');

        if(bracket != std::string::npos)
        {
            // Ignore cases where the last index is an array of struct index (SomeStruct[1].v should be ignored)
            if((dot == std::string::npos) || (bracket > dot))
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

    /** Copy text from a std::string to a char buffer, ensures null termination.
    */
    inline void copyStringToBuffer(char* buffer, uint32_t bufferSize, const std::string& s)
    {
        const uint32_t length = std::min(bufferSize - 1, (uint32_t)s.length());
        s.copy(buffer, length);
        buffer[length] = '\0';
    }

    /** Converts a size in bytes to a human readable string:
        - prints bytes (B) if size < 1000 bytes
        - prints kilobytes (KB) if size < 1000 kilobytes
        - prints megabytes (MB) if size < 1000 megabytes
        - prints gigabytes (GB) if size < 1000 gigabytes
        - otherwise prints terabytes (TB)
        \param[in] size Size in bytes
        \return Returns a human readable string.
    */
    inline std::string formatByteSize(size_t size)
    {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2);

        const size_t KB = 1000, MB = KB * KB, GB = MB * KB, TB = GB * KB;
        if (size < KB) oss << size << " B";
        else if (size < MB) oss << (size / (double)KB) << " KB";
        else if (size < GB) oss << (size / (double)MB) << " MB";
        else if (size < TB) oss << (size / (double)GB) << " GB";
        else oss << (size / (double)TB) << " TB";

        return oss.str();
    }

    /** Convert an ASCII string to a UTF-8 wstring
    */
    inline std::wstring string_2_wstring(const std::string& s)
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
        std::wstring ws = cvt.from_bytes(s);
        return ws;
    }

    /** Convert a UTF-8 wstring to an ASCII string
    */
    inline std::string wstring_2_string(const std::wstring& ws)
    {
        std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
        std::string s = cvt.to_bytes(ws);
        return s;
    }

    /** Convert a UTF-32 codepoint to a UTF-8 string
    */
    inline std::string utf32ToUtf8(uint32_t codepoint)
    {
#ifdef _WIN32
        std::wstring_convert<std::codecvt_utf8<uint32_t>, uint32_t> cvt;
#else
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cvt;
#endif
        return cvt.to_bytes(codepoint);
    }

    /** Encode data into base 64 encoding.
    */
    FALCOR_API std::string encodeBase64(const void* data, size_t len);

    /** Encode data into base 64 encoding.
    */
    inline std::string encodeBase64(const std::vector<uint8_t>& in)
    {
        return encodeBase64(in.data(), in.size());
    }

    /** Decode data from base 64 encoding.
    */
    FALCOR_API std::vector<uint8_t> decodeBase64(const std::string& in);

    /** Combine command line args to a single string
    */
    inline std::string concatCommandLine(uint32_t argc, char** argv)
    {
        std::string s;
        for (uint32_t i = 0; i < argc; i++)
        {
            s += std::string(argv[i]) + ((i < argc - 1) ? " " : "");
        }

        return s;
    }

    /*! @} */
};
