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
#include "SettingsUtils.h"
#include "Attributes.h"
#include "Core/Macros.h"
#include "Core/Error.h"
#include "Utils/Logger.h"

#include <type_traits>
#include <optional>
#include <regex>
#include <string>
#include <string_view>

#include <nlohmann/json.hpp>


namespace pybind11
{
class dict;
}

namespace Falcor
{

namespace settings
{

class AttributeFilter
{
    struct Record
    {
        std::string name;
        std::regex regex;
        nlohmann::json attributes;
    };

public:
    void add(const nlohmann::json& json) { addJson(json); }
    void clear() { mAttributes.clear(); }

    Attributes getAttributes(std::string_view shapeName_) const;

    template<typename T>
    std::optional<T> getAttribute(std::string_view shapeName, std::string_view attrName) const
    {
        nlohmann::json attribute = nullptr;

        for (const Record& recordIt : mAttributes)
        {
            if (!std::regex_match(shapeName.begin(), shapeName.end(), recordIt.regex))
                continue;
            auto attrIt = recordIt.attributes.find(attrName);
            if (attrIt != recordIt.attributes.end())
                attribute = attrIt.value();
        }

        if (attribute.is_null())
            return {};

        if (!detail::TypeChecker<T>::validType(attribute))
            throw detail::TypeError("Attribute's type does not match the requested type.");

        if constexpr (std::is_arithmetic_v<T>)
        {
            if (attribute.is_boolean())
                return attribute.get<bool>() ? T(1) : T(0);
        }

        return attribute.get<T>();
    }

    template<typename T>
    T getAttribute(std::string_view shapeName, std::string_view attrName, const T& def) const
    {
        auto result = getAttribute<T>(shapeName, attrName);
        return result ? *result : def;
    }

private:
    void addJson(const nlohmann::json& json);
    void addArray(const nlohmann::json& array);
    void addDictionary(const nlohmann::json& dict);

private:
    /// Filters out all attributes using the deprecated `name.filter` syntax,
    /// processes into filters, and returns the remaining attributes
    nlohmann::json processDeprecatedFilters(std::string_view name, nlohmann::json flattened, const std::string& regexStr);

private:
    std::vector<Record> mAttributes;
};

} // namespace settings

} // namespace Falcor
