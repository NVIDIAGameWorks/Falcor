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
#include "Core/Enum.h"
#include <pybind11/pytypes.h>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

namespace Falcor
{
class PythonDictionary
{
public:
    using Container = pybind11::dict;

    PythonDictionary() = default;
    PythonDictionary(const Container& c) : mMap(c) {}

    class Value
    {
    public:
        Value(const Container& container, const std::string_view name) : mName(name), mContainer(container){};
        Value(const Container& container = {}) : Value(container, std::string()) {}

        Value& operator=(const Value& rhs)
        {
            mName = rhs.mName;
            mContainer[mName.c_str()] = rhs.mContainer[mName.c_str()];
            return *this;
        }

        template<typename T>
        void operator=(const T& t)
        {
            if constexpr (has_enum_info_v<T>)
                mContainer[mName.c_str()] = enumToString(t);
            else
                mContainer[mName.c_str()] = t;
        }

        void operator=(const std::filesystem::path& path)
        {
            // Convert path to string. Otherwise it's represented as a Python WindowsPath/PosixPath.
            *this = path.generic_string();
        }

        template<typename T>
        operator T() const
        {
            if constexpr (has_enum_info_v<T>)
                return stringToEnum<T>(mContainer[mName.c_str()].cast<std::string>());
            else
                return mContainer[mName.c_str()].cast<T>();
        }

    private:
        std::string mName;
        const Container& mContainer;
    };

    template<typename ContainerType>
    class IteratorT
    {
    public:
        IteratorT(ContainerType* pContainer, const pybind11::detail::dict_iterator& it) : mIt(it), mpContainer(pContainer) {}

        bool operator==(const IteratorT& other) const { return other.mIt == mIt; }
        bool operator!=(const IteratorT& other) const { return other.mIt != mIt; }
        IteratorT& operator++()
        {
            mIt++;
            return *this;
        }
        IteratorT operator++(int)
        {
            ++mIt;
            return *this;
        }

        std::pair<std::string, Value> operator*()
        {
            std::string key = mIt->first.cast<std::string>();
            return {key, Value(*mpContainer, key)};
        }

    private:
        pybind11::detail::dict_iterator mIt;
        ContainerType* mpContainer;
    };

    using Iterator = IteratorT<Container>;
    using ConstIterator = IteratorT<const Container>;

    Value operator[](const std::string_view name) { return Value(mMap, name); }
    const Value operator[](const std::string_view name) const { return Value(mMap, name); }

    template<typename T>
    T get(const std::string_view name, const T& def) const
    {
        if (!mMap.contains(name.data()))
            return def;
        return mMap[name.data()].cast<T>();
    }

    template<typename T>
    std::optional<T> get(const std::string_view name) const
    {
        if (!mMap.contains(name.data()))
            return std::optional<T>();
        return std::optional<T>(mMap[name.data()].cast<T>());
    }

    // Avoid forcing std::string creation if Value would be just temporary
    template<typename T>
    T get(const std::string_view name) const
    {
        return mMap[name.data()].cast<T>();
    }

    ConstIterator begin() const { return ConstIterator(&mMap, mMap.begin()); }
    ConstIterator end() const { return ConstIterator(&mMap, mMap.end()); }

    Iterator begin() { return Iterator(&mMap, mMap.begin()); }
    Iterator end() { return Iterator(&mMap, mMap.end()); }

    size_t size() const { return mMap.size(); }

    bool keyExists(const std::string& key) const { return mMap.contains(key.c_str()); }

    pybind11::dict toPython() const { return mMap; }

    std::string toString() const { return pybind11::str(static_cast<pybind11::dict>(mMap)); }

private:
    Container mMap;
};
} // namespace Falcor
