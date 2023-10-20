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
#pragma once
#include "Core/Macros.h"
#include "Core/Error.h"
#include <algorithm>
#include <codecvt>
#include <iomanip>
#include <locale>
#include <string>
#include <vector>

namespace Falcor
{
/**
 * Check is a string starts with another string
 * @param[in] str String to check in
 * @param[in] prefix Prefix to check for
 * @param[in] caseSensitive Whether comparison should be case-sensitive
 * @return Returns true if string starts with the specified prefix.
 */
FALCOR_API bool hasPrefix(const std::string& str, const std::string& prefix, bool caseSensitive = true);

/**
 * Check is a string ends with another string
 * @param[in] str String to check in
 * @param[in] suffix Suffix to check for
 * @param[in] caseSensitive Whether comparison should be case-sensitive
 * @return Returns true if string ends with the specified suffix
 */
FALCOR_API bool hasSuffix(const std::string& str, const std::string& suffix, bool caseSensitive = true);

/**
 * Split a string into a vector of strings based on d delimiter
 * @param[in] str String to split
 * @param[in] delim Delimiter to split strings by
 * @return Array of split strings excluding delimiters.
 */
FALCOR_API std::vector<std::string> splitString(const std::string& str, const std::string& delim);

/**
 * Join an array of strings separated by another set string
 * @param[in] strings Array of strings to join.
 * @param[in] separator String placed between each string to be joined.
 * @return Joined string.
 */
FALCOR_API std::string joinStrings(const std::vector<std::string>& strings, const std::string& separator);

/**
 * Remove leading whitespace.
 * @param[in] str String to operate on.
 * @param[in] whitespace Whitespace characters.
 * @return String with leading whitespace removed.
 */
FALCOR_API std::string removeLeadingWhitespace(const std::string& str, const char* whitespace = " \n\r\t");

/**
 * Remove trailing whitespace.
 * @param[in] str String to operate on.
 * @param[in] whitespace Whitespace characters.
 * @return String with trailing whitespace removed.
 */
FALCOR_API std::string removeTrailingWhitespace(const std::string& str, const char* whitespace = " \n\r\t");

/**
 * Remove leading and trailing whitespace.
 * @param[in] str String to operate on.
 * @param[in] whitespace Whitespace characters.
 * @return String with leading and trailing whitespace removed.
 */
FALCOR_API std::string removeLeadingTrailingWhitespace(const std::string& str, const char* whitespace = " \n\r\t");

/**
 * Replace a set of character.
 * Example: replaceCharacters("some/path with/whitespace", "/ ", '_') returns "some_path_with_whitespace"
 * @param[in] str String to operate on.
 * @param[in] characters Set of characters to replace.
 * @param[in] replacement Character to use as a replacement.
 * @return String with characeters replaced.
 */
FALCOR_API std::string replaceCharacters(const std::string& str, const char* characters, const char replacement);

/**
 * Pad string to minimum length.
 */
FALCOR_API std::string padStringToLength(const std::string& str, size_t length, char padding = ' ');

/**
 * Replace all occurrences of a substring in a string. The function doesn't change the original string.
 * @param input The input string
 * @param src The substring to replace
 * @param dst The substring to replace Src with
 */
FALCOR_API std::string replaceSubstring(const std::string& input, const std::string& src, const std::string& dst);

/**
 * Decode an URI string.
 */
FALCOR_API std::string decodeURI(const std::string& input);

/**
 * Parses a string in the format <name>[<index>]. If format is valid, outputs the base name and the array index.
 * @param[in] name String to parse
 * @param[out] nonArray Becomes set to the non-array index portion of the string
 * @param[out] index Becomes set to the index value parsed from the string
 * @return Whether string was successfully parsed.
 */
FALCOR_API bool parseArrayIndex(const std::string& name, std::string& nonArray, uint32_t& index);

/**
 * Copy text from a std::string to a char buffer, ensures null termination.
 */
FALCOR_API void copyStringToBuffer(char* buffer, uint32_t bufferSize, const std::string& s);

/**
 * Converts a size in bytes to a human readable string:
 * - prints bytes (B) if size < 1000 bytes
 * - prints kilobytes (KB) if size < 1000 kilobytes
 * - prints megabytes (MB) if size < 1000 megabytes
 * - prints gigabytes (GB) if size < 1000 gigabytes
 * - otherwise prints terabytes (TB)
 * @param[in] size Size in bytes
 * @return Returns a human readable string.
 */
FALCOR_API std::string formatByteSize(size_t size);

/**
 * Convert an ASCII string to lower case.
 */
inline std::string toLowerCase(const std::string& str)
{
    std::string s = str;
    std::transform(str.begin(), str.end(), s.begin(), ::tolower);
    return s;
}

/**
 * Convert an ASCII string to a UTF-8 wstring
 */
inline std::wstring string_2_wstring(const std::string& s)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
    std::wstring ws = cvt.from_bytes(s);
    return ws;
}

/**
 * Convert a UTF-8 wstring to an ASCII string
 */
inline std::string wstring_2_string(const std::wstring& ws)
{
    std::wstring_convert<std::codecvt_utf8<wchar_t>> cvt;
    std::string s = cvt.to_bytes(ws);
    return s;
}

/**
 * Convert a UTF-32 codepoint to a UTF-8 string
 */
inline std::string utf32ToUtf8(uint32_t codepoint)
{
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cvt;
    return cvt.to_bytes(codepoint);
}

/**
 * Encode data into base 64 encoding.
 */
FALCOR_API std::string encodeBase64(const void* data, size_t len);

/**
 * Encode data into base 64 encoding.
 */
inline std::string encodeBase64(const std::vector<uint8_t>& in)
{
    return encodeBase64(in.data(), in.size());
}

/**
 * Decode data from base 64 encoding.
 */
FALCOR_API std::vector<uint8_t> decodeBase64(const std::string& in);
}; // namespace Falcor
