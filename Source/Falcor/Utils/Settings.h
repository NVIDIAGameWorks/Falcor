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

#include <nlohmann/json.hpp>

#include <regex>
#include <optional>
#include <mutex>
#include <map>
#include <filesystem>
#include <vector>

namespace pybind11
{
class dict;
}

namespace Falcor
{
class Settings;
class SettingsProperties
{
public:
    class TypeError : public std::runtime_error
    {
    public:
        using std::runtime_error::runtime_error;
    };

private:
    template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    static bool isType(const nlohmann::json& json)
    {
        return json.is_number() || json.is_boolean();
    }

    template<typename T, std::enable_if_t<std::is_same_v<T, std::string>, bool> = true>
    static bool isType(const nlohmann::json& json)
    {
        return json.is_string();
    }

    template<typename T, std::enable_if_t<std::is_same_v<T, SettingsProperties>, bool> = true>
    static bool isType(const nlohmann::json& json)
    {
        return json.is_object();
    }

    template<typename T, std::enable_if_t<std::is_same_v<T, nlohmann::json>, bool> = true>
    static bool isType(const nlohmann::json& json)
    {
        return true;
    }

    template<typename T, typename U, size_t N, std::enable_if_t<std::is_same_v<T, std::array<U, N>>, bool> = true>
    static bool isType(const nlohmann::json& json)
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
    struct JsonCaster
    {
        static T cast(const nlohmann::json& json)
        {
            if constexpr (std::is_arithmetic_v<T>)
            {
                if (json.is_boolean())
                    return json.get<bool>() ? T(1) : T(0);
            }
            return json.get<T>();
        }
    };

    template<typename gccfix>
    struct JsonCaster<bool, gccfix>
    {
        static bool cast(const nlohmann::json& json)
        {
            if (json.is_boolean())
                return json.get<bool>();
            return json.get<int>() != 0;
        }
    };

    template<typename gccfix>
    struct JsonCaster<SettingsProperties, gccfix>
    {
        static SettingsProperties cast(const nlohmann::json& json)
        {
            FALCOR_ASSERT(json.is_object());
            if (!json.is_object())
                throw TypeError("Trying to get SettingsProperties from a non-object json");
            return SettingsProperties(&json);
        }
    };

    template<typename gccfix>
    struct JsonCaster<nlohmann::json, gccfix>
    {
        static nlohmann::json cast(const nlohmann::json& json) { return json; }
    };

    template<typename U, size_t N, typename gccfix>
    struct JsonCaster<std::array<U, N>, gccfix>
    {
        using ArrayType = std::array<U, N>;
        static bool assertHelper(const nlohmann::json& json) { return isType<ArrayType, U, N>(json); }

        static ArrayType cast(const nlohmann::json& json)
        {
            FALCOR_ASSERT(assertHelper(json));

            ArrayType result;
            for (size_t i = 0; i < N; ++i)
                result[i] = JsonCaster<U>::cast(json[i]);
            return result;
        }
    };

public:
    SettingsProperties() = default;

    // Returns a property of the given name
    template<typename T>
    std::optional<T> get(const std::string_view name) const
    {
        // The underlying library requires attribute names to be null terminated
        FALCOR_ASSERT(!name.empty() && name.data()[name.size()] == 0);

        // Handle the : in names as nesting separator
        {
            std::string_view::size_type pos = name.find_first_of(':');
            if (pos != std::string_view::npos)
            {
                // The parent *must* be std::string, because the underlying pybind11
                // operates on const char*, so the length in string_view would be ignored,
                // and the whole name would be looked up, rather than just a subsection of it.

                // Name up to the :, excluding
                std::string parent(name.data(), pos);
                // Name after the :
                std::string_view child(name.data() + pos + 1, name.size() - pos - 1);
                if (!hasExact<SettingsProperties>(parent))
                    return std::optional<T>();
                return get<SettingsProperties>(parent)->get<T>(child);
            }
        }

        if (!hasExact(name))
            return std::optional<T>();

        try
        {
            return JsonCaster<T>::cast((*mDictionary)[name.data()]);
        }
        catch (const nlohmann::json::type_error& e)
        {
            throw TypeError(e.what());
        }
    }

    // Returns the property of the given name.
    // Throws TypeError if the type does not match
    template<typename T>
    T get(const std::string_view name, const T& def) const
    {
        // The underlying library requires attribute names to be null terminated
        FALCOR_ASSERT(!name.empty() && name.data()[name.size()] == 0);

        std::optional<T> opt = get<T>(name);
        return opt ? *opt : def;
    }

    // Do we have a property of the given name
    bool has(const std::string_view name) const
    {
        // The underlying library requires attribute names to be null terminated
        FALCOR_ASSERT(!name.empty() && name.data()[name.size()] == 0);

        // Handle the : in names as nesting separator
        {
            std::string_view::size_type pos = name.find_first_of(':');
            if (pos != std::string_view::npos)
            {
                // The parent *must* be std::string, because the underlying pybind11
                // operates on const char*, so the length in string_view would be ignored,
                // and the whole name would be looked up, rather than just a subsection of it.

                // Name up to the :, excluding
                std::string parent(name.data(), pos);
                // Name after the :
                std::string_view child(name.data() + pos + 1, name.size() - pos - 1);
                if (!hasExact<SettingsProperties>(parent))
                    return false;
                return get<SettingsProperties>(parent)->has(child);
            }
        }

        return hasExact(name);
    }

    // Do we have a property of the given name and type
    template<typename T>
    bool has(const std::string_view name) const
    {
        // The underlying library requires attribute names to be null terminated
        FALCOR_ASSERT(!name.empty() && name.data()[name.size()] == 0);

        // Handle the : in names as nesting separator
        {
            std::string_view::size_type pos = name.find_first_of(':');
            if (pos != std::string_view::npos)
            {
                // The parent *must* be std::string, because the underlying pybind11
                // operates on const char*, so the length in string_view would be ignored,
                // and the whole name would be looked up, rather than just a subsection of it.

                // Name up to the :, excluding
                std::string parent(name.data(), pos);
                // Name after the :
                std::string_view child(name.data() + pos + 1, name.size() - pos - 1);
                if (!hasExact<SettingsProperties>(parent))
                    return false;
                return get<SettingsProperties>(parent)->has<T>(child);
            }
        }

        return hasExact<T>(name);
    }

    std::string toString() const { return mDictionary->dump(); }

private:
    SettingsProperties(const nlohmann::json* dictionary) : mDictionary(dictionary) {}

    // Do we have a property of the given name
    bool hasExact(const std::string_view name) const
    {
        // The underlying library requires attribute names to be null terminated
        FALCOR_ASSERT(!name.empty() && name.data()[name.size()] == 0);

        return mDictionary->contains(name);
    }

    // Do we have a property of the given name and type
    template<typename T>
    bool hasExact(const std::string_view name) const
    {
        // The underlying library requires attribute names to be null terminated
        FALCOR_ASSERT(!name.empty() && name.data()[name.size()] == 0);

        if (!has(name))
            return false;

        return isType<T>((*mDictionary)[name.data()]);
    }

    friend class Settings;

private:
    const nlohmann::json* mDictionary{nullptr};
};

class FALCOR_API Settings
{
public:
    using SearchDirectories = std::vector<std::filesystem::path>;

    /// Get the global settings instance.
    static Settings& getGlobalSettings();

    Settings() : mData(1) {}

    SettingsProperties getOptions() const { return SettingsProperties(&getActive().mOptions); }

    template<typename T>
    std::optional<T> getOption(const std::string_view name) const
    {
        return SettingsProperties(&getActive().mOptions).get<T>(name);
    }

    template<typename T>
    T getOption(const std::string_view name, const T& def) const
    {
        return SettingsProperties(&getActive().mOptions).get(name, def);
    }

    // Adds to global option list.
    // It is a nested list of dictionaries
    void addOptions(const pybind11::dict& options);

    /// Add options from a JSON file, returning true on success and false on failure
    bool addOptions(const std::filesystem::path& path);

    // Clears the global options to defaults
    void clearOptions() { getActive().mOptions.clear(); }

    /**
     * Attributes don't really belong here. They should be part of the loadScene.
     * However, if you load scenes from the GUI, you want the attributes to get applied to all the scenes
     * (think setting attributes to "make all curves tessellate fine").
     *
     * So this, which is effectively an attribute filter (RIF, if you will) need to live in some
     * reasonably global place. This will be replaced with a more principled solution in the new Scene.
     */

    template<typename T>
    std::optional<T> getAttribute(const std::string_view shapeName, const std::string_view attributeName) const
    {
        // The underlying library requires attribute names to be null terminated
        FALCOR_ASSERT(!attributeName.empty() && attributeName.data()[attributeName.size()] == 0);

        SettingsProperties props(&getActive().mFilteredAttributes);

        std::string str = props.toString();

        // Don't have it at all (or have it with a wrong type)
        if (!props.has<T>(attributeName))
            return std::optional<T>();

        std::string attributeNameFilter = std::string(attributeName) + ".filter";

        // If we don't have the filter at all, we can return the value right away
        if (!props.has(attributeNameFilter))
            return props.get<T>(attributeName);

        // filter can be either std::string, or a pybind11::list of 1 or 2 values
        // (first being std::string, second optionally boolean).
        // Unfortunately, pybind makes us do a ton of unnecessary lookups here:
        std::string expression;
        bool negateRegex = false;
        if (props.has<std::string>(attributeNameFilter))
        {
            expression = props.get(attributeNameFilter, expression);
        }
        else
        {
            nlohmann::json array = *props.get<nlohmann::json>(attributeNameFilter);
            FALCOR_CHECK(array.is_array(), "Expecting an array");
            expression = array[0].get<std::string>();
            if (array.size() > 1)
                negateRegex = array[1].get<bool>();
        }

        if (matches(shapeName, expression) == negateRegex)
            return std::optional<T>();

        return props.get<T>(attributeName);
    }

    template<typename T>
    T getAttribute(const std::string_view shapeName, const std::string_view attributeName, const T& def) const
    {
        // The underlying library requires attribute names to be null terminated
        FALCOR_ASSERT(!attributeName.empty() && attributeName.data()[attributeName.size()] == 0);

        auto opt = getAttribute<T>(shapeName, attributeName);
        return opt ? *opt : def;
    }

    // This is a global way to add per-object attributes.
    // Again, follows a key-value mapping, logic, but adds a logic for attaching the
    // values to objects by name. This is only/mostly useful for the Tiger demo,
    // where the Tiger asset gets a different behavior than the others.
    //
    // The value is always an array of either 1, 2 or 3 items.
    // The first item is the assigned value.
    // The second is a "basic" C++ regex regular expression (POSIX),
    // and the value is applied to all objects whose name matches.
    // If the second value is not present, the Attribute is applied to all objects
    // If the second value is present, the third value (bool) says whether to negate
    // the expression. If false (default), Attribute is applied to all names that do match,
    // if true, it is applied to all names that do not match.
    //
    // Important, these generally kick in only during loading/processing of the scene.
    // The changes are NOT retroactive (e.g., if you load a scene and then disable motion,
    // it won't do anything unless you unload and reload the scene).
    //
    // The values are not combined in any smart way, the last value is applied,
    // e.g., you cannot use two subsequent calls to combine the regex in any way.
    // Reason: This is not really meant to replace DCC, just to get us out of the hot water
    //
    // Example usage:
    // m.addFilteredAttributes({'usdImporter':{'enableMotion':False}}); // disables motion on everything
    // or
    // m.addFilteredAttributes({'usdImporter':{'enableMotion':False, 'enableMotion.filter':['/World/Tiger/.*']}}); // disables motion on
    // Tiger only or m.addFilteredAttributes({'usdImporter':{'enableMotion':False, 'enableMotion.filter':['/World/Tiger/.*', False]}}); //
    // disables motion on all but Tiger
    //
    // Important, these two consequently
    // m.addFilteredAttributes({'usdImporter:enableMotion':[False]});
    // m.addFilteredAttributes({'usdImporter:enableMotion':[False, '/World/Tiger/.*']});
    // will *NOT* disable motion on everything and then re-enable it for the tiger,
    // they will only set the "enableMotion = False" on the Tiger, while leaving
    // everything else to default (which happens to be "False" as well)
    void addFilteredAttributes(const pybind11::dict& attributes);

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
        auto it = mSearchDirectories.find(std::string(category));
        if (it != mSearchDirectories.end())
            return it->second;
        it = mStandardSearchDirectories.find(std::string(category));
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
        nlohmann::json mOptions;
        nlohmann::json mFilteredAttributes;
    };

    SettingsData& getActive() { return mData.back(); }
    const SettingsData& getActive() const { return mData.back(); }

    bool addOptionsJSON(const std::filesystem::path& path);
    bool addOptionsTOML(const std::filesystem::path& path);

    static bool matches(const std::string_view shapeName, const std::string_view expression)
    {
        std::regex attrRegex(expression.data(), expression.size());
        std::smatch match;
        return std::regex_match(shapeName.data(), attrRegex);
    }

    static void deep_merge(nlohmann::json& lhs, const nlohmann::json& rhs)
    {
        FALCOR_ASSERT(lhs.is_object() && rhs.is_object());

        for (auto& rhsIt : rhs.items())
        {
            auto lhsIt = lhs.find(rhsIt.key());
            // Both are dictionaries, so we need to recurse (lhs will be passed by reference, but that reference is dictionary)
            if (lhsIt != lhs.end() && lhsIt.value().is_object() && rhsIt.value().is_object())
            {
                deep_merge(lhsIt.value(), rhsIt.value());
                continue;
            }
            lhs[rhsIt.key()] = rhsIt.value();
        }
    }

    static void merge(nlohmann::json& lhsJson, const nlohmann::json& rhsJson)
    {
        if (rhsJson.is_null())
            return;
        if (lhsJson.is_null())
        {
            lhsJson = rhsJson;
            return;
        }
        deep_merge(lhsJson, rhsJson);
    }

    void updateSearchPaths(const nlohmann::json& update);

private:
    std::vector<SettingsData> mData;

    std::map<std::string, SearchDirectories> mStandardSearchDirectories;
    std::map<std::string, SearchDirectories> mSearchDirectories;
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
