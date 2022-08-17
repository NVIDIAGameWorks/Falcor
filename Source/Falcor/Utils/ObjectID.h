/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "Core/Assert.h"
#include <fmt/format.h>
#include <pybind11/pybind11.h>

#include <limits>
#include <functional>
#include <type_traits>

namespace Falcor
{

/** Universal class for strongly typed IDs. Takes an TKindEnum to allow usage for IDs in unrelated
    subsystems without polluting a single enum with unrelated kinds.
    TODO: Ideally it would also take name of the ID for python binding purposes, but passing
    string as a template argument is non-trivial in C++17.

    \param TKindEnum Enum class from which kinds are drawn. Different enum classes are not directly convertible.
    \param TKind Kind of the ID.
    \param TIntType the underlying numeric type. It is advised that it should be the same for all TKinds in the same enum.
  */
template<typename TKindEnum, TKindEnum TKind, typename TIntType>
class ObjectID
{
public:
    using IntType = TIntType;
    static constexpr TKindEnum kKind = TKind;
    static constexpr IntType kInvalidID = std::numeric_limits<IntType>::max();

public:
    /** Default construction creates an invalid ID.
        TODO: Consider creating uninitialized ID instead, if vectors of IDs become a performance issue.
      */
    ObjectID()
        : mID(kInvalidID)
    {}

    /** Constructs ObjectID from any numeric type.
        Checks for validity of the ID with respect to th allowed range.

        \param[in] id Integer ID to initialize from.
      */
    template<typename T>
    explicit ObjectID(const T& id, std::enable_if_t<std::is_integral_v<T>, bool> = true)
        : mID(IntType(id))
    {
        // First we make sure it is positive
        FALCOR_ASSERT_GE(id, T(0));
        // When we know for a fact it is not negative,
        // we can cast it to the unsigned version of that integer for comparison
        // (otherwise compiler complains about signed/unsigned mismatch when entering literals)
        FALCOR_ASSERT_LE(std::make_unsigned_t<T>(id), std::numeric_limits<IntType>::max());
    }

    /** Allows converting between different Kinds of the same EnumKind.
        This is slightly safer than going straight through numeric ids via get().
        This is mostly used when converting from an "union" ID, that can identify different
        objects based on other flags, e.g., kCurveOrMesh that is either Curve or Mesh,
        based on the tessellation flags.
        NB: Ideally this would be removed, use as sparingly as possible.

        \param[in] other The ObjectID to be converted from.
      */
    template<TKindEnum TOtherKind>
    explicit ObjectID( const ObjectID<TKindEnum, TOtherKind, TIntType>& other )
    {
        mID = other.get();
    }

    /** A helper method when convering from an numeric ID in Slang, to the strongly typed CPU ID.
        This is separate from the a basic constructor only for the purpose of clearly identifying,
        the conversion in the code, as per Joel's "Making Wrong Code Look Wrong" principle.
        TODO: Remove once the slang side also has strongly typed IDs.

        \param[in] id Integer ID to initialize from
        \return The ObjectID created from a numeric ID
      */
    template<typename T>
    static ObjectID fromSlang(const T& id, std::enable_if_t<std::is_integral_v<T>, bool> = true)
    {
        return ObjectID{ id };
    }

    /** Provides an invalid ID for comparison purposes.
        In the future, most uses would be replaced by either isValid (for comparison),
        or by ObjectID(ObjectID::kInvalidID) (for obtaining an invalid ID)

        \return An invalid ObjectID.
      */
    static ObjectID Invalid()
    {
        return ObjectID();
    }

    /** Returns true when the ID is valid, i.e., get() != kInvalidID

        \return True when valid.
      */
    bool isValid() const
    {
        return mID != kInvalidID;
    }

    /** Return the numeric value of the ID.
        Should be used rather sparingly, e.g., consider allowing objects to be indexed by the strongly
        typed ID, instead of just a number.
        NB: Consider using getters with strongly typed IDs, rather than directly accessing even vectors/buffers.

        \return Numeric value of the ID, can be kInvalidID.
      */
    IntType get() const
    {
        return mID;
    }

    /** A helped method to convert to numeric ID in Slang. Functionally identical to get(),
        but in the future it should be removed, and the Slang should have a compatible and checked
        strongly typed ID as well. Separated from get() to clearly show all such locations.

        \return Numeric value of the ID, can be kInvalidID.
      */
    IntType getSlang() const
    {
        return get();
    }

    bool operator==(const ObjectID& rhs) const
    {
        return mID == rhs.mID;
    }

    bool operator!=(const ObjectID& rhs) const
    {
        return mID != rhs.mID;
    }

    bool operator<=(const ObjectID& rhs) const
    {
        return mID <= rhs.mID;
    }

    bool operator>=(const ObjectID& rhs) const
    {
        return mID >= rhs.mID;
    }

    bool operator<(const ObjectID& rhs) const
    {
        return mID < rhs.mID;
    }

    bool operator>(const ObjectID& rhs) const
    {
        return mID < rhs.mID;
    }

    ObjectID& operator++()
    {
        FALCOR_ASSERT_LT(mID, std::numeric_limits<IntType>::max());
        ++mID;
        return *this;
    }

    ObjectID operator++(int)
    {
        return ObjectID(mID++);
    }
private:
    IntType mID;
};

template<typename TKindEnum, TKindEnum TKind, typename TIntType>
inline std::string to_string(const ObjectID<TKindEnum, TKind, TIntType>& v)
{
    return std::to_string(v.get());
}

}

template<typename TKindEnum, TKindEnum TKind, typename TIntType>
struct std::hash<Falcor::ObjectID<TKindEnum, TKind, TIntType>>
{
    using ObjectID = Falcor::ObjectID<TKindEnum, TKind, TIntType>;
    std::size_t operator()(const ObjectID& id) const noexcept
    {
        return std::hash<typename ObjectID::IntType>{}(id.get());
    }
};

template<typename TKindEnum, TKindEnum TKind, typename TIntType>
struct fmt::formatter<Falcor::ObjectID<TKindEnum, TKind, TIntType>>
{
    using ObjectID = Falcor::ObjectID<TKindEnum, TKind, TIntType>;

    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(const ObjectID& id, FormatContext& ctx)
    {
        return fmt::format_to(ctx.out(), "{0}", id.get());
    }
};

namespace pybind11::detail
{
    template<typename TKindEnum, TKindEnum TKind, typename TIntType>
    struct type_caster<Falcor::ObjectID<TKindEnum, TKind, TIntType>>
    {
        using ObjectID = Falcor::ObjectID<TKindEnum, TKind, TIntType>;
    public:
        PYBIND11_TYPE_CASTER(ObjectID, const_name("ObjectID"));

        bool load(handle src, bool)
        {
            PyObject* source = src.ptr();
            PyObject* tmp = PyNumber_Long(source);
            if (!tmp)
                return false;

            typename ObjectID::IntType idValue = PyLong_AsUnsignedLong(tmp);
            Py_DECREF(tmp);

            value = (idValue == ObjectID::kInvalidID) ? ObjectID() : ObjectID(idValue);
            return !PyErr_Occurred();
        }

        static handle cast(const ObjectID& src, return_value_policy /* policy */, handle /* parent */)
        {
            return PyLong_FromUnsignedLong(src.get());
        }
    };
}
