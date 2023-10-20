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

#include "Error.h"
#include "Utils/Logger.h"
#include <fstd/span.h>
#include <string>
#include <string_view>
#include <type_traits>
#include <algorithm>
#include <vector>
#include <fmt/core.h>

namespace Falcor
{
// Helper using ADL to find EnumInfo in other namespaces.
template<typename T>
using EnumInfo = decltype(falcorFindEnumInfoADL(std::declval<T>()));

template<typename, typename = void>
struct has_enum_info : std::false_type
{};

template<typename T>
struct has_enum_info<T, std::void_t<decltype(EnumInfo<T>::items)>> : std::true_type
{};

template<typename T>
inline constexpr bool has_enum_info_v = has_enum_info<T>::value;

/**
 * Convert an enum value to a string.
 * Throws if the enum value is not found in the registered enum information.
 */
template<typename T, std::enable_if_t<has_enum_info_v<T>, bool> = true>
inline const std::string& enumToString(T value)
{
    const auto& items = EnumInfo<T>::items();
    auto it = std::find_if(items.begin(), items.end(), [value](const auto& item) { return item.first == value; });
    if (it == items.end())
        FALCOR_THROW("Invalid enum value {}", int(value));
    return it->second;
}

/**
 * Convert a string to an enum value.
 * Throws if the string is not found in the registered enum information.
 */
template<typename T, std::enable_if_t<has_enum_info_v<T>, bool> = true>
inline T stringToEnum(std::string_view name)
{
    const auto& items = EnumInfo<T>::items();
    auto it = std::find_if(items.begin(), items.end(), [name](const auto& item) { return item.second == name; });
    if (it == items.end())
        FALCOR_THROW("Invalid enum name '{}'", name);
    return it->first;
}

/**
 * Check if an enum has a value with the given name.
 */
template<typename T, std::enable_if_t<has_enum_info_v<T>, bool> = true>
inline bool enumHasValue(std::string_view name)
{
    const auto& items = EnumInfo<T>::items();
    auto it = std::find_if(items.begin(), items.end(), [name](const auto& item) { return item.second == name; });
    return it != items.end();
}

/**
 * Convert an flags enum value to a list of strings.
 * Throws if any of the flags are not found in the registered enum information.
 */
template<typename T, std::enable_if_t<has_enum_info_v<T>, bool> = true>
inline std::vector<std::string> flagsToStringList(T flags)
{
    std::vector<std::string> list;
    const auto& items = EnumInfo<T>::items();
    for (const auto& item : items)
    {
        if (is_set(flags, item.first))
        {
            list.push_back(item.second);
            flip_bit(flags, item.first);
        }
    }
    if (flags != T(0))
        FALCOR_THROW("Invalid enum flags value {}", int(flags));
    return list;
}

/**
 * Convert a list of strings to a flags enum value.
 * Throws if any of the strings are not found in the registered enum information.
 */
template<typename T, std::enable_if_t<has_enum_info_v<T>, bool> = true>
inline T stringListToFlags(const std::vector<std::string>& list)
{
    T flags = T(0);
    for (const auto& name : list)
        flags |= stringToEnum<T>(name);
    return flags;
}

} // namespace Falcor

/**
 * Define enum information. This is expected to be used as follows:
 *
 * enum class Foo { A, B, C };
 * FALCOR_ENUM_INFO(Foo, {
 *     { Foo::A, "A" },
 *     { Foo::B, "B" },
 *     { Foo::C, "C" },
 * })
 */
#define FALCOR_ENUM_INFO(T, ...)                                    \
    struct T##_info                                                 \
    {                                                               \
        static fstd::span<std::pair<T, std::string>> items()        \
        {                                                           \
            static std::pair<T, std::string> items[] = __VA_ARGS__; \
            return {std::begin(items), std::end(items)};            \
        }                                                           \
    };

/**
 * Register enum information to be used with helper functions.
 * This needs to be placed outside of any structs but within the
 * namespace of the enum:
 *
 * namespace ns
 * {
 * struct Bar
 * {
 *     enum class Foo { A, B, C };
 *     FALCOR_ENUM_INFO(Foo, ...)
 * };
 *
 * FALCOR_ENUM_REGISTER(Bar::Foo)
 * } // namespace ns
 *
 * Registered enums can be converted to/from strings using:
 * - enumToString<Enum>(Enum value)
 * - stringToEnum<Enum>(std::string_view name)
 */
#define FALCOR_ENUM_REGISTER(T)                                            \
    constexpr T##_info falcorFindEnumInfoADL [[maybe_unused]] (T) noexcept \
    {                                                                      \
        return T##_info{};                                                 \
    }

/// Enum formatter.
template<typename T>
struct fmt::formatter<T, std::enable_if_t<Falcor::has_enum_info_v<T>, char>> : formatter<std::string>
{
    template<typename FormatContext>
    auto format(const T& e, FormatContext& ctx)
    {
        return formatter<std::string>::format(Falcor::enumToString(e), ctx);
    }
};
