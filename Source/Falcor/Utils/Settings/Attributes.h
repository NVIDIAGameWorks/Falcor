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
#include "SettingsUtils.h"
#include "Core/Macros.h"
#include "Core/Error.h"

#include <type_traits>
#include <optional>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

namespace Falcor
{
namespace settings
{

class Attributes
{
public:
    Attributes() = default;
    Attributes(nlohmann::json jsonDict) : mJsonDict(jsonDict) {}

    void overrideWith(const Attributes& other) { addDict(other.mJsonDict); }

    template<typename T>
    std::optional<T> get(std::string_view attrName) const
    {
        nlohmann::json attribute = nullptr;

        auto attrIt = mJsonDict.find(attrName);
        if (attrIt != mJsonDict.end())
            attribute = attrIt.value();

        if (attribute.is_null())
            return {};

        if (!detail::TypeChecker<T>::validType(attribute))
            throw detail::TypeError("Attribute's type does not match the requested type.");

        // Handle return value of bool, if the actual is convertible to bool (from int, usually)
        if constexpr (std::is_same_v<T, bool>)
        {
            if (attribute.is_boolean())
                return attribute.get<bool>();
            return attribute.get<int>() != 0;
        }

        // Handle return value of int, if the actual is a boolean
        if constexpr (std::is_arithmetic_v<T>)
        {
            if (attribute.is_boolean())
                return attribute.get<bool>() ? T(1) : T(0);
        }

        return attribute.get<T>();
    }

    template<typename T>
    T get(std::string_view attrName, const T& def) const
    {
        auto result = get<T>(attrName);
        return result ? *result : def;
    }

    bool has(std::string_view attrName) const
    {
        auto attrIt = mJsonDict.find(attrName);
        return attrIt != mJsonDict.end();
    }

    void addDict(nlohmann::json jsonDict)
    {
        for (auto& it : jsonDict.items())
            mJsonDict[it.key()] = it.value();
    }

    void clear() { mJsonDict = nlohmann::json::object(); }

    void removePrefix(std::string_view prefix)
    {
        nlohmann::json filtered;
        for (auto& it : mJsonDict.items())
        {
            if (it.key().find(prefix) != 0)
                filtered[it.key()] = it.value();
        }
        mJsonDict = std::move(filtered);
    }

    void removeExact(std::string_view name) { mJsonDict.erase(name); }

    std::string to_string() const { return mJsonDict.dump(); }

private:
    nlohmann::json mJsonDict;
};

} // namespace settings
} // namespace Falcor
