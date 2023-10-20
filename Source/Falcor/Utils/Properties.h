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
#include "Core/Enum.h"
#include "Utils/Math/VectorTypes.h"

// Do not include the full "nlohmann/json.hpp" header here, it will increase compile time considerably.
#include <nlohmann/json_fwd.hpp>
#include <pybind11/pytypes.h>
#include <fmt/core.h>

#include <memory>
#include <optional>
#include <string_view>
#include <string>
#include <type_traits>
#include <filesystem>
#include <cstdint>

namespace Falcor
{

class Properties;

namespace detail
{

/// Primary template handles all types not supporting the operation.
template<class T, template<typename> class, typename = std::void_t<>>
struct detect : std::false_type
{};

/// Specialization recognizes/validates only types supporting the archetype.
template<class T, template<typename> class Op>
struct detect<T, Op, std::void_t<Op<T>>> : std::true_type
{};

class DummyArchive
{};

template<typename T>
using serialize_t = decltype(&T::template serialize<DummyArchive>);

/// Check if the type T has a void serialize<Archive>(Archive&) member function template.
template<typename T>
inline constexpr bool has_serialize_v = detect<T, serialize_t>::value;

} // namespace detail

template<typename T>
Properties serializeToProperties(const T& value);

template<typename T>
T deserializeFromProperties(const Properties& props);

/**
 * A class for storing properties.
 *
 * Properties are stored as a JSON object. The JSON object is ordered, so the order of properties is preserved.
 * Using JSON as a backing storage, properties can easily be serialized to/from files.
 * This class also supports conversion to/from python dictionaries, making it easy to specify properties from python.
 *
 * For usage patterns, look at the unit tests.
 */
class FALCOR_API Properties
{
public:
    using json = nlohmann::ordered_json;

    Properties();
    Properties(const json& j);
    Properties(json&& j);
    Properties(const pybind11::dict& d);

    Properties(const Properties& other);
    Properties(Properties&& other);

    ~Properties();

    Properties& operator=(const Properties& other);
    Properties& operator=(Properties&& other);

    /// Converts the properties to a JSON object.
    json toJson() const;

    /// Converts the properties to a python dictionary.
    pybind11::dict toPython() const;

    /// Dumps the properties to a string.
    std::string dump(int indent = -1) const;

    /// Check if the properties are empty.
    bool empty() const;

    /// Check if a property exists.
    bool has(std::string_view name) const;

    /// Set a property.
    template<typename T>
    void set(std::string_view name, const T& value)
    {
        if constexpr (has_enum_info_v<T>)
        {
            setInternal(name, enumToString(value));
        }
        else if constexpr (detail::has_serialize_v<T>)
        {
            setInternal(name, serializeToProperties(value));
        }
        else
        {
            setInternal<T>(name, value);
        }
    }

    /// Get a property.
    /// Throws if property does not exist or has the wrong type.
    template<typename T>
    T get(std::string_view name) const
    {
        if constexpr (has_enum_info_v<T>)
        {
            std::string value;
            if (!getInternal(name, value))
                FALCOR_THROW("Property '{}' does not exist.", name);
            return stringToEnum<T>(value);
        }
        else if constexpr (detail::has_serialize_v<T>)
        {
            Properties props;
            if (!getInternal(name, props))
                FALCOR_THROW("Property '{}' does not exist.", name);
            return deserializeFromProperties<T>(props);
        }
        else
        {
            T value;
            if (!getInternal(name, value))
                FALCOR_THROW("Property '{}' does not exist.", name);
            return value;
        }
    }

    /// Get a property.
    /// Returns the default value if the property does not exist.
    /// Throws if the property exists but has the wrong type.
    template<typename T>
    T get(std::string_view name, const T& def) const
    {
        if constexpr (has_enum_info_v<T>)
        {
            std::string value;
            if (!getInternal(name, value))
                return def;
            return stringToEnum<T>(value);
        }
        else if constexpr (detail::has_serialize_v<T>)
        {
            Properties props;
            if (!getInternal(name, props))
                return def;
            return deserializeFromProperties<T>(props);
        }
        else
        {
            T value;
            if (!getInternal(name, value))
                return def;
            return value;
        }
    }

    /// Get a property.
    /// Stores the value to the passed reference and returns true if it exists.
    /// Returns false otherwise.
    /// Throws if the property exists but has the wrong type.
    template<typename T>
    bool getTo(std::string_view name, T& value) const
    {
        if constexpr (has_enum_info_v<T>)
        {
            std::string enumString;
            bool result = getInternal(name, enumString);
            if (result)
                value = stringToEnum<T>(enumString);
            return result;
        }
        else if constexpr (detail::has_serialize_v<T>)
        {
            Properties props;
            bool result = getInternal(name, props);
            if (result)
                value = deserializeFromProperties<T>(props);
            return result;
        }
        else
        {
            return getInternal<T>(name, value);
        }
    }

    /// Get a property.
    /// Returns empty optional if the property does not exist.
    /// Throws if the property exists but has the wrong type.
    template<typename T>
    std::optional<T> getOpt(std::string_view name) const
    {
        if constexpr (has_enum_info_v<T>)
        {
            std::string enumString;
            bool result = getInternal(name, enumString);
            return result ? std::make_optional<T>(stringToEnum<T>(enumString)) : std::nullopt;
        }
        else if constexpr (detail::has_serialize_v<T>)
        {
            Properties props;
            bool result = getInternal(name, props);
            return result ? std::make_optional<T>(deserializeFromProperties<T>(props)) : std::nullopt;
        }
        else
        {
            T value;
            bool result = getInternal<T>(name, value);
            return result ? std::make_optional<T>(value) : std::nullopt;
        }
    }

    /// Convenience overload for handling C strings.
    void set(std::string_view name, const char* value) { set(name, std::string(value)); }

    // --------------------------------------------------------------------
    // Comparison
    // --------------------------------------------------------------------

    bool operator==(const Properties& rhs) const;
    bool operator!=(const Properties& rhs) const;

    // --------------------------------------------------------------------
    // Accessors and iterators
    // --------------------------------------------------------------------

    /// Value accessor.
    class Value
    {
    public:
        template<typename T>
        void operator=(const T& value) const
        {
            mProperties.set(mName, value);
        }

        template<typename T>
        operator T() const
        {
            return mProperties.get<T>(mName);
        }

    private:
        Value(Properties& properties, std::string_view name) : mProperties(properties), mName(name) {}
        Properties& mProperties;
        std::string mName;

        friend class Properties;
    };

    /// Constant value accessor.
    class ConstValue
    {
    public:
        template<typename T>
        operator T() const
        {
            return mProperties.get<T>(mName);
        }

    private:
        ConstValue(const Properties& properties, std::string_view name) : mProperties(properties), mName(name) {}
        const Properties& mProperties;
        std::string mName;

        friend class Properties;
    };

    /// Iterator.
    class FALCOR_API Iterator
    {
    public:
        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;
        Iterator& operator++();
        std::pair<std::string, Value> operator*();
        ~Iterator();

    private:
        struct Impl;
        Iterator(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> mImpl;
        friend class Properties;
    };

    /// Constant iterator.
    class FALCOR_API ConstIterator
    {
    public:
        bool operator==(const ConstIterator& other) const;
        bool operator!=(const ConstIterator& other) const;
        ConstIterator& operator++();
        std::pair<std::string, ConstValue> operator*();
        ~ConstIterator();

    private:
        struct Impl;
        ConstIterator(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> mImpl;
        friend class Properties;
    };

    Value operator[](std::string_view name) { return Value(*this, name); }
    const ConstValue operator[](std::string_view name) const { return ConstValue(*this, name); }

    Iterator begin();
    Iterator end();

    ConstIterator begin() const;
    ConstIterator end() const;

private:
    template<typename T>
    void setInternal(std::string_view name, const T& value);

    template<typename T>
    bool getInternal(std::string_view name, T& value) const;

    std::unique_ptr<json> mJson;
};

#define EXTERN_PROPERTY_ACCESSOR(T)                                                         \
    extern template FALCOR_API void Properties::setInternal<T>(std::string_view, const T&); \
    extern template FALCOR_API bool Properties::getInternal<T>(std::string_view, T&) const;

EXTERN_PROPERTY_ACCESSOR(bool)
EXTERN_PROPERTY_ACCESSOR(int32_t)
EXTERN_PROPERTY_ACCESSOR(int64_t)
EXTERN_PROPERTY_ACCESSOR(uint32_t)
EXTERN_PROPERTY_ACCESSOR(uint64_t)
EXTERN_PROPERTY_ACCESSOR(float)
EXTERN_PROPERTY_ACCESSOR(double)
EXTERN_PROPERTY_ACCESSOR(std::string)
EXTERN_PROPERTY_ACCESSOR(std::filesystem::path)
EXTERN_PROPERTY_ACCESSOR(int2)
EXTERN_PROPERTY_ACCESSOR(int3)
EXTERN_PROPERTY_ACCESSOR(int4)
EXTERN_PROPERTY_ACCESSOR(uint2)
EXTERN_PROPERTY_ACCESSOR(uint3)
EXTERN_PROPERTY_ACCESSOR(uint4)
EXTERN_PROPERTY_ACCESSOR(float2)
EXTERN_PROPERTY_ACCESSOR(float3)
EXTERN_PROPERTY_ACCESSOR(float4)
EXTERN_PROPERTY_ACCESSOR(Properties)

#undef EXTERN_PROPERTY_ACCESSOR

/**
 * Helper class for serializing objects into Properties.
 */
class PropertiesWriter
{
public:
    template<typename T>
    void operator()(std::string_view name, const T& value)
    {
        if constexpr (detail::has_serialize_v<T>)
            mProperties.set<Properties>(name, PropertiesWriter::write<T>(value));
        else
            mProperties.set<T>(name, value);
    }

    template<typename T>
    static Properties write(const T& value)
    {
        PropertiesWriter writer;
        const_cast<T&>(value).serialize(writer);
        return writer.mProperties;
    }

private:
    PropertiesWriter() = default;
    Properties mProperties;
};

/**
 * Helper class for deserializing objects from Properties.
 */
class PropertiesReader
{
public:
    template<typename T>
    void operator()(std::string_view name, T& value)
    {
        if constexpr (detail::has_serialize_v<T>)
        {
            Properties props;
            if (mProperties.getTo<Properties>(name, props))
                value = PropertiesReader::read<T>(props);
        }
        else
            mProperties.getTo<T>(name, value);
    }

    template<typename T>
    static T read(const Properties& props)
    {
        T value;
        PropertiesReader reader(props);
        value.serialize(reader);
        return value;
    }

private:
    PropertiesReader(const Properties& props) : mProperties(props) {}
    const Properties& mProperties;
};

template<typename T>
Properties serializeToProperties(const T& value)
{
    return PropertiesWriter::write(value);
}

template<typename T>
T deserializeFromProperties(const Properties& props)
{
    return PropertiesReader::read<T>(props);
}

} // namespace Falcor

template<>
struct fmt::formatter<Falcor::Properties> : formatter<std::string>
{
    template<typename FormatContext>
    auto format(const Falcor::Properties& props, FormatContext& ctx) const
    {
        return formatter<std::string>::format(props.dump(), ctx);
    }
};
