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

class TypeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace settings::detail
} // namespace Falcor
