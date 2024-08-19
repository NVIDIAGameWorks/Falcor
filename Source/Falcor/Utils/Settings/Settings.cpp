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
#include "Settings.h"
#include "Utils/StringUtils.h"
#include "Utils/PathResolving.h"

#include "Utils/Scripting/ScriptBindings.h"

#include <pybind11/pybind11.h>
#include <pybind11_json/pybind11_json.hpp>

#include <fstream>

namespace Falcor
{

namespace
{

std::vector<std::string> toStrings(const nlohmann::json& value)
{
    std::vector<std::string> result;
    if (value.is_string())
    {
        result.push_back(value.get<std::string>());
        return result;
    }
    else if (value.is_array())
    {
        for (size_t i = 0; i < value.size(); ++i)
        {
            if (value[i].is_string())
                result.push_back(value[i].get<std::string>());
        }
    }

    return result;
}

} // namespace

Settings& Settings::getGlobalSettings()
{
    static Settings globalSettings = []()
    {
        Settings settings;
        // Load settings from runtime directory first.
        settings.addOptions(getRuntimeDirectory() / "settings.json");
        // Override with user settings.
        if (!getHomeDirectory().empty())
            settings.addOptions(getHomeDirectory() / ".falcor" / "settings.json");
        return settings;
    }();
    return globalSettings;
}

void Settings::addOptions(const nlohmann::json& options)
{
    nlohmann::json flattened = settings::detail::flattenDictionary(options);
    for (auto& it : flattened.items())
        getActive().mOptions.removePrefix(it.key());
    getActive().mOptions.addDict(flattened);
    updateSearchPaths(flattened);
}

void Settings::addOptions(const pybind11::dict& options)
{
    addOptions(settings::detail::flattenDictionary(pyjson::to_json(options)));
}

void Settings::addOptions(const pybind11::list& options)
{
    addOptions(settings::detail::flattenDictionary(pyjson::to_json(options)));
}

bool Settings::addOptions(const std::filesystem::path& path)
{
    if (path.extension() != ".json")
        return false;

    if (!std::filesystem::exists(path))
        return false;
    std::ifstream ifs(path);
    if (!ifs)
        return false;
    nlohmann::json jf = settings::detail::flattenDictionary(
        nlohmann::json::parse(ifs, nullptr /*callback*/, true /*allow exceptions*/, true /*ignore comments*/)
    );
    for (auto& it : jf.items())
        getActive().mOptions.removePrefix(it.key());
    getActive().mOptions.addDict(jf);
    updateSearchPaths(jf);
    return true;
}

void Settings::addFilteredAttributes(const pybind11::dict& attributes)
{
    addFilteredAttributes(pyjson::to_json(attributes));
}

void Settings::addFilteredAttributes(const pybind11::list& attributes)
{
    addFilteredAttributes(pyjson::to_json(attributes));
}

void Settings::addFilteredAttributes(const nlohmann::json& attributes)
{
    FALCOR_CHECK(attributes.is_array() || attributes.is_object(), "The attributes must be a dictionary, or an array of dictionaries.");
    getActive().mAttributeFilters.add(attributes);
}

bool Settings::addFilteredAttributes(const std::filesystem::path& path)
{
    if (path.extension() != ".json")
        return false;

    if (!std::filesystem::exists(path))
        return false;
    std::ifstream ifs(path);
    if (!ifs)
        return false;

    nlohmann::json jf = nlohmann::json::parse(ifs, nullptr /*callback*/, true /*allow exceptions*/, true /*ignore comments*/);

    getActive().mAttributeFilters.add(jf);
    return true;
}

void Settings::clearFilteredAttributes()
{
    getActive().mAttributeFilters.clear();
}

void Settings::updateSearchPaths(const nlohmann::json& update)
{
    if (update.is_null() || !update.is_object())
        return;
    for (auto& updateIt : update.items())
    {
        auto processPath = [this](std::string_view searchKind, std::string_view category, const std::vector<std::string>& pathUpdates)
        {
            if (pathUpdates.empty())
                return;

            if (searchKind == "standardsearchpath")
            {
                std::vector<std::filesystem::path>& current = mStandardSearchDirectories[std::string(category)];
                ResolvedPaths result = resolveSearchPaths(current, pathUpdates, std::vector<std::filesystem::path>());
                FALCOR_CHECK(
                    result.invalid.empty(),
                    "While processing {}:{}, found invalid paths: {}",
                    searchKind,
                    category,
                    joinStrings(result.invalid, ", ")
                );
                current = std::move(result.resolved);
                return;
            }

            if (searchKind == "searchpath")
            {
                std::vector<std::filesystem::path>& current = mSearchDirectories[std::string(category)];
                auto it = mStandardSearchDirectories.find(std::string(category));

                ResolvedPaths result;
                if (it == mStandardSearchDirectories.end())
                    result = resolveSearchPaths(current, pathUpdates, std::vector<std::filesystem::path>());
                else
                    result = resolveSearchPaths(current, pathUpdates, it->second);
                FALCOR_CHECK(
                    result.invalid.empty(),
                    "While processing {}:{}, found invalid paths: {}",
                    searchKind,
                    category,
                    joinStrings(result.invalid, ", ")
                );
                current = std::move(result.resolved);
            }
        };

        if ((updateIt.key() == "searchpath" || updateIt.key() == "standardsearchpath") && updateIt.value().is_object())
        {
            for (auto& kindIt : updateIt.value().items())
            {
                std::vector<std::string> pathUpdates = toStrings(kindIt.value());
                processPath(updateIt.key(), kindIt.key(), pathUpdates);
            }
            continue;
        }

        // Check for `searchpath:foo` or `standardsearchpath:foo`
        std::string_view searchKind;
        std::string_view category;

        if (updateIt.key().find("searchpath:") == 0)
        {
            searchKind = "searchpath";
            category = std::string_view(updateIt.key()).substr(std::strlen("searchpath:"));
        }

        if (updateIt.key().find("standardsearchpath:") == 0)
        {
            searchKind = "standardsearchpath";
            category = std::string_view(updateIt.key()).substr(std::strlen("standardsearchpath:"));
        }

        if (searchKind.empty())
            continue;

        std::vector<std::string> pathUpdates = toStrings(updateIt.value());
        processPath(searchKind, category, pathUpdates);
    }
}

FALCOR_SCRIPT_BINDING(Settings)
{
    using namespace pybind11::literals;

    pybind11::class_<Settings> settings(m, "Settings");
    settings.def("addOptions", pybind11::overload_cast<const pybind11::dict&>(&Settings::addOptions), "dict"_a);
    settings.def("addFilteredAttributes", pybind11::overload_cast<const pybind11::dict&>(&Settings::addFilteredAttributes), "dict"_a);
    settings.def("addFilteredAttributes", pybind11::overload_cast<const pybind11::list&>(&Settings::addFilteredAttributes), "list"_a);
    settings.def("clearOptions", &Settings::clearOptions);
    settings.def("clearFilteredAttributes", &Settings::clearFilteredAttributes);
}

} // namespace Falcor
