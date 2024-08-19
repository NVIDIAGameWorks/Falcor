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
#include "AttributeFilters.h"
#include "Core/Macros.h"
#include "Core/Error.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <map>
#include <filesystem>
#include <vector>

namespace pybind11
{
class dict;
class list;
} // namespace pybind11

namespace Falcor
{

class FALCOR_API Settings
{
public:
    using Options = settings::Attributes;
    using Attributes = settings::Attributes;
    using TypeError = settings::detail::TypeError;

public:
    using SearchDirectories = std::vector<std::filesystem::path>;

    /// Get the global settings instance.
    static Settings& getGlobalSettings();

    Settings() : mData(1) {}

    const Options& getOptions() const { return getActive().mOptions; }

    template<typename T>
    std::optional<T> getOption(std::string_view optionName) const
    {
        return getActive().mOptions.get<T>(optionName);
    }

    template<typename T>
    T getOption(std::string_view optionName, const T& def) const
    {
        return getActive().mOptions.get(optionName, def);
    }

    // Adds to global option list.
    // It is a nested list of dictionaries
    void addOptions(const nlohmann::json& options);
    void addOptions(const pybind11::dict& options);
    void addOptions(const pybind11::list& options);

    /// Add options from a JSON file, returning true on success and false on failure
    bool addOptions(const std::filesystem::path& path);

    // Clears the global options to defaults
    void clearOptions() { getActive().mOptions.clear(); }

    /**
     * The Settings object now contains Attribute filters, that are applied once by one
     * to a shape based on it name, to provide the final Attribute.
     */

    template<typename T>
    std::optional<T> getAttribute(std::string_view shapeName, std::string_view attributeName) const
    {
        return getActive().mAttributeFilters.getAttribute<T>(shapeName, attributeName);
    }

    template<typename T>
    T getAttribute(std::string_view shapeName, std::string_view attributeName, const T& def) const
    {
        return getActive().mAttributeFilters.getAttribute<T>(shapeName, attributeName, def);
    }

    Attributes getAttributes(std::string_view shapeName) const { return getActive().mAttributeFilters.getAttributes(shapeName); }

    /**
     * @brief Adds filtered attributes with the following syntax.
     *
     * Examples use the JSON with comments syntax.
     * Full example, applies "foo":4 and "bar:foobar":5 to all shapes that start with "Fur":
     * {
     *     // optional name of the filter, not currently used
     *     "name" : "name of the filter",
     *     // Optional regex that determines to what shape names will the attributes apply.
     *     // Will be ".*" (apply to all) if not present.
     *     "regex" : "Fur.*",
     *     // Attributes that will be applied. Can be nested dictionaries, but will be flattened to colon
     *     // separated names, e.g.: { "foo" : 4, "bar:foobar" : 5 }
     *     "attributes" :
     *     {
     *         "foo" : 4
     *         "bar" :
     *         {
     *             "foobar" : 5
     *         }
     *     }
     * }
     *
     * Simplified example, applies "foo":6 to every shape:
     * {
     *     // If "attributes" is not part of the dictionary, the whole dictionary is taken as attributes block
     *     "foo":6
     * }
     *
     * Can also set multiple such filters at once, as an array:
     * [ { dict1 }, { dict2 } ]
     *
     * Attributes are applied in the order in which they were added.
     *
     * Can still correctly process the deprecated syntax with .filter name:
     * {
     *     // apply "foo":4 to shapes matching regex
     *     "foo":4
     *     "foo.filter": "<regex>"
     *     // apply "bar":5 to all shapes not matching the regex
     *     "bar":5
     *     "bar.filter": ["<regex>", true]
     * }
     *
     * @param attributes Dictionary or array of dictionaries that provides attributes and their filters
     */
    void addFilteredAttributes(const nlohmann::json& attributes);
    void addFilteredAttributes(const pybind11::dict& attributes);
    void addFilteredAttributes(const pybind11::list& attributes);
    bool addFilteredAttributes(const std::filesystem::path& path);

    // Clears all the attributes to default
    void clearFilteredAttributes();

    /**
     * Returns search paths from the given category.
     *
     * If the searchpath is not defined, it returns the standardsearch path.
     * If that's not defined either, it returns an empty result.
     *
     * @param category
     * @return
     */
    const SearchDirectories& getSearchDirectories(std::string_view category)
    {
        auto it = mSearchDirectories.find(category);
        if (it != mSearchDirectories.end())
            return it->second;
        it = mStandardSearchDirectories.find(category);
        if (it != mStandardSearchDirectories.end())
            return it->second;

        static SearchDirectories sEmpty; // TODO: REMOVEGLOBAL
        return sEmpty;
    }

    void pushSettingsStack() { mData.push_back(mData.back()); }

    void popSettingsStack()
    {
        mData.pop_back();
        FALCOR_ASSERT(!mData.empty());
    }

private:
    struct SettingsData
    {
        Options mOptions;
        settings::AttributeFilter mAttributeFilters;
    };

    SettingsData& getActive() { return mData.back(); }
    const SettingsData& getActive() const { return mData.back(); }

    void updateSearchPaths(const nlohmann::json& update);

private:
    std::vector<SettingsData> mData;

    std::map<std::string, SearchDirectories, std::less<>> mStandardSearchDirectories;
    std::map<std::string, SearchDirectories, std::less<>> mSearchDirectories;
};

class ScopedSettings
{
public:
    ScopedSettings(Settings& settings) : mSettings(settings) { mSettings.pushSettingsStack(); }

    ~ScopedSettings() { mSettings.popSettingsStack(); }

private:
    Settings& mSettings;
};

} // namespace Falcor
