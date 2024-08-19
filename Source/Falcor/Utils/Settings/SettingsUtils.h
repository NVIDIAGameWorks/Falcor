/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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

#include <exception>
#include <string>
#include <type_traits>
#include <array>

#include <nlohmann/json.hpp>

#include <fmt/format.h>

namespace Falcor
{
namespace settings::detail
{

namespace
{
void flattenDictionary(const nlohmann::json& dict, const std::string& prefix, nlohmann::json& flattened)
{
    FALCOR_ASSERT(flattened.is_object());
    if (!dict.is_object())
    {
        flattened[prefix] = dict;
        return;
    }

    for (auto& it : dict.items())
    {
        std::string name = fmt::format("{}{}{}", prefix, prefix.empty() ? "" : ":", it.key());
        flattenDictionary(it.value(), name, flattened);
    }
}
} // namespace

/// Flattens nested dictionaries into colon separated name,
/// e.g. {"foo":{"bar":4}} becomes {"foo:bar":4}
inline nlohmann::json flattenDictionary(const nlohmann::json& dict)
{
    nlohmann::json flattened = nlohmann::json::object();
    flattenDictionary(dict, "", flattened);
    return flattened;
}

template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
inline bool isType(const nlohmann::json& json)
{
    return json.is_number() || json.is_boolean();
}

template<typename T, std::enable_if_t<std::is_same_v<T, std::string>, bool> = true>
inline bool isType(const nlohmann::json& json)
{
    return json.is_string();
}

template<typename T, typename U, size_t N, std::enable_if_t<std::is_same_v<T, std::array<U, N>>, bool> = true>
inline bool isType(const nlohmann::json& json)
{
    if (!json.is_array())
        return false;
    if (json.size() != N)
        return false;
    for (size_t i = 0; i < N; ++i)
        if (!isType<U>(json[i]))
            return false;
    return true;
}

// The "gccfix" parameter is used to avoid "explicit specialization in non-namespace scope" in gcc.
// See https://gcc.gnu.org/bugzilla/show_bug.cgi?id=85282
template<typename T, typename gccfix = void>
struct TypeChecker
{
    static bool validType(const nlohmann::json& json) { return isType<T>(json); }
};

template<typename U, size_t N, typename gccfix>
struct TypeChecker<std::array<U, N>, gccfix>
{
    using ArrayType = std::array<U, N>;
    static bool validType(const nlohmann::json& json) { return isType<ArrayType, U, N>(json); }
};

class TypeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace settings::detail
} // namespace Falcor
