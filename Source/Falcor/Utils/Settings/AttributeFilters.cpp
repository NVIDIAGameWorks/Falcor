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
#include "AttributeFilters.h"

#include <set>

namespace Falcor
{
namespace settings
{
Attributes AttributeFilter::getAttributes(std::string_view shapeName) const
{
    Attributes result;
    for (const Record& recordIt : mAttributes)
    {
        if (!std::regex_match(shapeName.begin(), shapeName.end(), recordIt.regex))
            continue;

        result.addDict(recordIt.attributes);
    }

    return result;
}

void AttributeFilter::addJson(const nlohmann::json& json)
{
    if (json.is_array())
        addArray(json);
    else
        addDictionary(json);
}

void AttributeFilter::addArray(const nlohmann::json& array)
{
    for (auto& arrayIt : array)
        addDictionary(arrayIt);
}

void AttributeFilter::addDictionary(const nlohmann::json& dict)
{
    FALCOR_CHECK(dict.is_object(), "Must be object/dictionary.");

    auto nameIt = dict.find("name");
    auto regexIt = dict.find("regex");
    auto attrIt = dict.find("attributes");

    FALCOR_CHECK(regexIt == dict.end() || regexIt.value().is_string(), "Filter `regex` must have string value.");
    FALCOR_CHECK(nameIt == dict.end() || nameIt.value().is_string(), "Filter `name` must have string value.");
    FALCOR_CHECK(attrIt == dict.end() || attrIt.value().is_object(), "`Filter attributes` have dictionary/object value.");
    FALCOR_CHECK(
        regexIt == dict.end() || (regexIt != dict.end() && attrIt != dict.end()),
        "If `filter` is present, `attributes` must be present as well."
    );

    Record record;

    record.name = fmt::format("filter_{}", mAttributes.size());
    if (nameIt != dict.end())
        record.name = nameIt.value().get<std::string>();

    std::string regexStr = ".*";
    if (regexIt != dict.end())
        regexStr = regexIt.value().get<std::string>();
    record.regex = std::regex(regexStr);

    nlohmann::json allFlattened;
    if (attrIt != dict.end())
        allFlattened = detail::flattenDictionary(attrIt.value());
    else
        allFlattened = detail::flattenDictionary(dict);

    // filter out the .filter of the old attributes
    record.attributes = processDeprecatedFilters(record.name, allFlattened, regexStr);

    if (!record.attributes.empty())
        mAttributes.push_back(std::move(record));
}

/// Filters out all attributes using the deprecated `name.filter` syntax,
/// processes into filters, and returns the remaining attributes
nlohmann::json AttributeFilter::processDeprecatedFilters(std::string_view name, nlohmann::json flattened, const std::string& regexStr)
{
    std::set<std::string> processed;
    for (auto& filterIt : flattened.items())
    {
        const std::string& filterKey = filterIt.key();
        size_t pos = filterKey.find(".filter");
        if (pos != filterKey.size() - 7)
            continue;

        std::string attrKey = filterKey.substr(0, pos);
        auto attrIT = flattened.find(attrKey);
        FALCOR_CHECK(attrIT != flattened.end(), "Found an attribute filter `{}`, but not the actual attribute `{}`.", filterKey, attrKey);
        FALCOR_CHECK(
            regexStr == ".*",
            "Found a filtered attribute `{}` found along with regex `{}`. Filtered attributes are only supported when regex is "
            "`.*`.",
            attrKey,
            regexStr
        );

        {
            std::string filterRegexStr;
            bool isNegatedRegex = false;
            if (filterIt.value().is_string())
            {
                filterRegexStr = filterIt.value().get<std::string>();
            }
            else
            {
                FALCOR_CHECK(
                    filterIt.value().is_array() && filterIt.value().size() <= 2 && filterIt.value()[0].is_string() &&
                        (filterIt.value().size() == 1 || filterIt.value()[1].is_boolean()),
                    "`{}` must be either a string or [<string>, <bool>] array.",
                    filterKey
                );
                filterRegexStr = filterIt.value()[0].get<std::string>();
                if (filterIt.value().size() == 2)
                    isNegatedRegex = filterIt.value()[1].get<bool>();
            }

            if (!isNegatedRegex)
            {
                Record filteredRecord;
                filteredRecord.name = fmt::format("{}_{}", name, filterKey);
                filteredRecord.regex = std::regex(filterRegexStr);
                filteredRecord.attributes = nlohmann::json::object();
                filteredRecord.attributes[attrIT.key()] = attrIT.value();
                mAttributes.push_back(std::move(filteredRecord));
            }
            else
            {
                Record filteredRecord;
                filteredRecord.name = fmt::format("{}_{}_apply", name, filterKey);
                filteredRecord.regex = std::regex(".*");
                filteredRecord.attributes = nlohmann::json::object();
                filteredRecord.attributes[attrIT.key()] = attrIT.value();
                mAttributes.push_back(std::move(filteredRecord));

                filteredRecord.name = fmt::format("{}_{}_unapply", name, filterKey);
                filteredRecord.regex = std::regex(filterRegexStr);
                filteredRecord.attributes = nlohmann::json::object();
                filteredRecord.attributes[attrIT.key()] = nullptr;
                mAttributes.push_back(std::move(filteredRecord));
            }
        }

        processed.insert(filterKey);
        processed.insert(attrKey);
    }

    if (processed.empty())
        return flattened;

    nlohmann::json result = nlohmann::json::object();
    for (auto& filterIt : flattened.items())
    {
        if (processed.count(filterIt.key()) > 0)
            continue;
        result[filterIt.key()] = filterIt.value();
    }

    return result;
}

} // namespace settings
} // namespace Falcor
