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
#include "Properties.h"

#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl/filesystem.h>

namespace Falcor
{
using json = Properties::json;

namespace
{
template<typename VecT>
json vecToJson(const VecT& vec)
{
    auto constexpr length = VecT::length();
    auto j = json::array();
    for (size_t i = 0; i < length; ++i)
        j.push_back(vec[i]);
    return j;
}

template<typename T>
json valueToJson(const T& value)
{
    if constexpr (std::is_same_v<T, Properties>)
    {
        return value.toJson();
    }
    else if constexpr (std::is_same_v<T, int2> || std::is_same_v<T, int3> || std::is_same_v<T, int4>)
    {
        return vecToJson(value);
    }
    else if constexpr (std::is_same_v<T, uint2> || std::is_same_v<T, uint3> || std::is_same_v<T, uint4>)
    {
        return vecToJson(value);
    }
    else if constexpr (std::is_same_v<T, float2> || std::is_same_v<T, float3> || std::is_same_v<T, float4>)
    {
        return vecToJson(value);
    }
    else
    {
        return json(value);
    }
}

template<typename VecT>
VecT vecFromJson(const json& json, std::string_view name)
{
    auto constexpr length = VecT::length();
    if (!json.is_array())
        FALCOR_THROW("Property '{}' is not an array.", name);
    if (json.size() != length)
        FALCOR_THROW("Property '{}' has an invalid number of elements.", name);
    VecT result;
    for (size_t i = 0; i < length; ++i)
        json[i].get_to(result[i]);
    return result;
}

template<typename T>
T valueFromJson(const json& json, std::string_view name)
{
    if constexpr (std::is_same_v<T, bool>)
    {
        if (!json.is_boolean())
            FALCOR_THROW("Property '{}' is not a boolean.", name);
        return static_cast<T>(json);
    }
    else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>)
    {
        if (!json.is_number_integer())
            FALCOR_THROW("Property '{}' is not an integer.", name);
        return static_cast<T>(json);
    }
    else if constexpr (std::is_integral_v<T> && !std::is_signed_v<T>)
    {
        // if (!json.is_number_intenger() !json.is_number_unsigned()) FALCOR_THROW("Property '{}' is not an unsigned integer.", name);
        return static_cast<T>(json);
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        // Allow integers to be converted to floating point
        if (!json.is_number_float() && !json.is_number_integer())
            FALCOR_THROW("Property '{}' is not a floating point value or integer.", name);
        return static_cast<T>(json);
    }
    else if constexpr (std::is_same_v<T, std::string>)
    {
        if (!json.is_string())
            FALCOR_THROW("Property '{}' is not a string.", name);
        return static_cast<T>(json);
    }
    else if constexpr (std::is_same_v<T, std::filesystem::path>)
    {
        if (!json.is_string())
            FALCOR_THROW("Property '{}' is not a string/path.", name);
        return static_cast<std::string>(json);
    }
    else if constexpr (std::is_same_v<T, Properties>)
    {
        if (!json.is_object())
            FALCOR_THROW("Property '{}' is not an object.", name);
        return Properties(json);
    }
    else if constexpr (std::is_same_v<T, int2> || std::is_same_v<T, int3> || std::is_same_v<T, int4>)
    {
        return vecFromJson<T>(json, name);
    }
    else if constexpr (std::is_same_v<T, uint2> || std::is_same_v<T, uint3> || std::is_same_v<T, uint4>)
    {
        return vecFromJson<T>(json, name);
    }
    else if constexpr (std::is_same_v<T, float2> || std::is_same_v<T, float3> || std::is_same_v<T, float4>)
    {
        return vecFromJson<T>(json, name);
    }
}

json pythonToJson(const pybind11::handle& obj)
{
    if (obj.ptr() == nullptr || obj.is_none())
    {
        return nullptr;
    }
    if (pybind11::isinstance<pybind11::bool_>(obj))
    {
        return obj.cast<bool>();
    }
    if (pybind11::isinstance<pybind11::int_>(obj))
    {
        try
        {
            json::number_integer_t s = obj.cast<json::number_integer_t>();
            if (pybind11::int_(s).equal(obj))
            {
                return s;
            }
        }
        catch (...)
        {}
        try
        {
            json::number_unsigned_t u = obj.cast<json::number_unsigned_t>();
            if (pybind11::int_(u).equal(obj))
            {
                return u;
            }
        }
        catch (...)
        {}
        throw std::runtime_error(
            "pythonToJson received an integer out of range for both json::number_integer_t and json::number_unsigned_t type: " +
            pybind11::repr(obj).cast<std::string>()
        );
    }
    if (pybind11::isinstance<pybind11::float_>(obj))
    {
        return obj.cast<double>();
    }
    // if (pybind11::isinstance<pybind11::bytes>(obj))
    // {
    //     pybind11::module base64 = pybind11::module::import("base64");
    //     return base64.attr("b64encode")(obj).attr("decode")("utf-8").cast<std::string>();
    // }
    if (pybind11::isinstance<pybind11::str>(obj))
    {
        return obj.cast<std::string>();
    }
    if (pybind11::isinstance<pybind11::tuple>(obj) || pybind11::isinstance<pybind11::list>(obj))
    {
        auto out = json::array();
        for (const pybind11::handle value : obj)
        {
            out.push_back(pythonToJson(value));
        }
        return out;
    }
    if (pybind11::isinstance<pybind11::dict>(obj))
    {
        auto out = json::object();
        for (const pybind11::handle key : obj)
        {
            out[pybind11::str(key).cast<std::string>()] = pythonToJson(obj[key]);
        }
        return out;
    }
    // isinstance<std::filesystem::path> doesn't work for all PathLike so we also check the type name.
    const char* tp_name = obj.ptr()->ob_type->tp_name;
    if (pybind11::isinstance<std::filesystem::path>(obj) || std::strcmp(tp_name, "WindowsPath") == 0 ||
        std::strcmp(tp_name, "PosixPath") == 0)
    {
        return obj.cast<std::filesystem::path>().string();
    }
#define VEC_TO_JSON(T)                   \
    if (pybind11::isinstance<T>(obj))    \
    {                                    \
        return vecToJson(obj.cast<T>()); \
    }
    VEC_TO_JSON(int2)
    VEC_TO_JSON(int3)
    VEC_TO_JSON(int4)
    VEC_TO_JSON(uint2)
    VEC_TO_JSON(uint3)
    VEC_TO_JSON(uint4)
    VEC_TO_JSON(float2)
    VEC_TO_JSON(float3)
    VEC_TO_JSON(float4)
    throw std::runtime_error("pythonToJson not implemented for this type of object: " + pybind11::repr(obj).cast<std::string>());
}

pybind11::object jsonToPython(const json& j)
{
    if (j.is_null())
    {
        return pybind11::none();
    }
    else if (j.is_boolean())
    {
        return pybind11::bool_(j.get<bool>());
    }
    else if (j.is_number_unsigned())
    {
        return pybind11::int_(j.get<json::number_unsigned_t>());
    }
    else if (j.is_number_integer())
    {
        return pybind11::int_(j.get<json::number_integer_t>());
    }
    else if (j.is_number_float())
    {
        return pybind11::float_(j.get<double>());
    }
    else if (j.is_string())
    {
        return pybind11::str(j.get<std::string>());
    }
    else if (j.is_array())
    {
        pybind11::list obj(j.size());
        for (std::size_t i = 0; i < j.size(); i++)
        {
            obj[i] = jsonToPython(j[i]);
        }
        return std::move(obj);
    }
    else // Object
    {
        pybind11::dict obj;
        for (json::const_iterator it = j.cbegin(); it != j.cend(); ++it)
        {
            obj[pybind11::str(it.key())] = jsonToPython(it.value());
        }
        return std::move(obj);
    }
}
} // namespace

Properties::Properties()
{
    mJson = std::make_unique<json>(json::object());
}

Properties::Properties(const json& j)
{
    mJson = std::make_unique<json>(j);
}

Properties::Properties(json&& j)
{
    mJson = std::make_unique<json>(std::move(j));
}

Properties::Properties(const pybind11::dict& d)
{
    mJson = std::make_unique<json>(pythonToJson(d));
}

Properties::Properties(const Properties& other)
{
    mJson = std::make_unique<json>(*other.mJson);
}

Properties::Properties(Properties&& other)
{
    mJson = std::move(other.mJson);
}

Properties::~Properties() {}

Properties& Properties::operator=(const Properties& other)
{
    mJson = std::make_unique<json>(*other.mJson);
    return *this;
}

Properties& Properties::operator=(Properties&& other)
{
    mJson = std::move(other.mJson);
    return *this;
}

json Properties::toJson() const
{
    return *mJson;
}

pybind11::dict Properties::toPython() const
{
    return jsonToPython(*mJson);
}

std::string Properties::dump(int indent) const
{
    return mJson->dump(indent);
}

bool Properties::empty() const
{
    return mJson->empty();
}

bool Properties::has(std::string_view name) const
{
    return mJson->find(name) != mJson->end();
}

template<typename T>
void Properties::setInternal(std::string_view name, const T& value)
{
    (*mJson)[name] = valueToJson<T>(value);
}

template<typename T>
bool Properties::getInternal(std::string_view name, T& value) const
{
    if (auto it = mJson->find(name); it != mJson->end())
    {
        value = valueFromJson<T>(*it, name);
        return true;
    }
    else
        return false;
}

bool Properties::operator==(const Properties& rhs) const
{
    return *mJson == *rhs.mJson;
}

bool Properties::operator!=(const Properties& rhs) const
{
    return !(*this == rhs);
}

// ------------------------------------------------------------------
// Iterator
// ------------------------------------------------------------------

struct Properties::Iterator::Impl
{
    Properties& properties;
    json::iterator it;
};

Properties::Iterator::Iterator(std::unique_ptr<Impl> impl) : mImpl(std::move(impl)) {}
Properties::Iterator::~Iterator() {}

bool Properties::Iterator::operator==(const Iterator& other) const
{
    return mImpl->it == other.mImpl->it;
}

bool Properties::Iterator::operator!=(const Iterator& other) const
{
    return mImpl->it != other.mImpl->it;
}

Properties::Iterator& Properties::Iterator::operator++()
{
    mImpl->it++;
    return *this;
}

std::pair<std::string, Properties::Value> Properties::Iterator::operator*()
{
    return std::make_pair(mImpl->it.key(), Value(mImpl->properties, mImpl->it.key()));
}

struct Properties::ConstIterator::Impl
{
    const Properties& properties;
    json::const_iterator it;
};

Properties::ConstIterator::ConstIterator(std::unique_ptr<Impl> impl) : mImpl(std::move(impl)) {}
Properties::ConstIterator::~ConstIterator() {}

bool Properties::ConstIterator::operator==(const ConstIterator& other) const
{
    return mImpl->it == other.mImpl->it;
}

bool Properties::ConstIterator::operator!=(const ConstIterator& other) const
{
    return mImpl->it != other.mImpl->it;
}

Properties::ConstIterator& Properties::ConstIterator::operator++()
{
    mImpl->it++;
    return *this;
}

std::pair<std::string, Properties::ConstValue> Properties::ConstIterator::operator*()
{
    return std::make_pair(mImpl->it.key(), ConstValue(mImpl->properties, mImpl->it.key()));
}

Properties::Iterator Properties::begin()
{
    return Iterator(std::make_unique<Iterator::Impl>(Iterator::Impl{*this, mJson->begin()}));
}

Properties::Iterator Properties::end()
{
    return Iterator(std::make_unique<Iterator::Impl>(Iterator::Impl{*this, mJson->end()}));
}

Properties::ConstIterator Properties::begin() const
{
    return ConstIterator(std::make_unique<ConstIterator::Impl>(ConstIterator::Impl{*this, mJson->begin()}));
}

Properties::ConstIterator Properties::end() const
{
    return ConstIterator(std::make_unique<ConstIterator::Impl>(ConstIterator::Impl{*this, mJson->end()}));
}

#define EXPORT_PROPERTY_ACCESSOR(T)                                                  \
    template FALCOR_API void Properties::setInternal<T>(std::string_view, const T&); \
    template FALCOR_API bool Properties::getInternal<T>(std::string_view, T&) const;

EXPORT_PROPERTY_ACCESSOR(bool)
EXPORT_PROPERTY_ACCESSOR(int32_t)
EXPORT_PROPERTY_ACCESSOR(int64_t)
EXPORT_PROPERTY_ACCESSOR(uint32_t)
EXPORT_PROPERTY_ACCESSOR(uint64_t)
EXPORT_PROPERTY_ACCESSOR(float)
EXPORT_PROPERTY_ACCESSOR(double)
EXPORT_PROPERTY_ACCESSOR(std::string)
EXPORT_PROPERTY_ACCESSOR(std::filesystem::path)
EXPORT_PROPERTY_ACCESSOR(int2)
EXPORT_PROPERTY_ACCESSOR(int3)
EXPORT_PROPERTY_ACCESSOR(int4)
EXPORT_PROPERTY_ACCESSOR(uint2)
EXPORT_PROPERTY_ACCESSOR(uint3)
EXPORT_PROPERTY_ACCESSOR(uint4)
EXPORT_PROPERTY_ACCESSOR(float2)
EXPORT_PROPERTY_ACCESSOR(float3)
EXPORT_PROPERTY_ACCESSOR(float4)
EXPORT_PROPERTY_ACCESSOR(Properties)

#undef EXPORT_PROPERTY_ACCESSOR

} // namespace Falcor
