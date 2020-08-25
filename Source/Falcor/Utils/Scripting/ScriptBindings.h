/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "pybind11/stl.h"

namespace Falcor::ScriptBindings
{
    using RegisterBindingFunc = std::function<void(pybind11::module& m)>;

    /** Register a script binding function.
        This function will be called when scripting is initialized.
        \param[in] f Function to be called for registering script bindings.
    */
    dlldecl void registerBinding(RegisterBindingFunc f);

    /************************************************************************/
    /* Helpers                                                              */
    /************************************************************************/

    /** Adds binary and/or operators to a Python enum.
        This allows the enum to be used as a set of flags instead of just a list of choices.
        \param[in] e Enum to be extended.
    */
    template<typename T>
    static void addEnumBinaryOperators(pybind11::enum_<T>& e)
    {
        e.def("__and__", [](const T& value1, const T& value2) { return T(int(value1) & int(value2)); });
        e.def("__or__", [](const T& value1, const T& value2) { return T(int(value1) | int(value2)); });
    }

    /** Returns the string representation of a value of a registered type.
        \param[in] value Value to be converted to a string.
        \return Returns the string representation.
    */
    template<typename T>
    static std::string repr(const T& value)
    {
        return pybind11::repr(pybind11::cast(value));
    }

    /** This helper allows to register bindings for simple data structs.
        This is accomplished by adding a __init__ (constructor) and a __repr__ implementation.
        The __init__ function takes kwargs and populates all the structs fields that have been
        registered with the field() method on this helper. The __repr__ implementation prints all
        the fields registered with the field() method. Lets assume we have a C++ struct:

        struct Example
        {
            int foo;
            std::string bar;
        };

        We can register bindings using:

        SerializableStruct<Example> example(m, "Example");
        example.field("foo", &Example::foo);
        example.field("bar", &Example::bar);

        In Python, we can then use the constructor like this:

        example = Example(foo=123, bar="test")

        Also, to serialize the instance into a string we can use repr:

        repr(example)

        which gives back a string like: Example(foo=123, bar="test")
    */
    template<typename T, typename... Options>
    struct SerializableStruct : public pybind11::class_<T, Options...>
    {
        using This = SerializableStruct<T, Options...>;

        static_assert(std::is_default_constructible_v<T> && std::is_copy_constructible_v<T>);

        template <typename... Extra>
        SerializableStruct(pybind11::handle scope, const char* name, const Extra&... extra)
            : pybind11::class_<T, Options...>(scope, name, extra...)
        {
            This::info().name = name;
            auto initFunc = [](const pybind11::kwargs& args) { return This::init(args); };
            this->def(pybind11::init(initFunc));
            this->def(pybind11::init<>());
            this->def("__repr__", This::repr);
        }

        template<typename D, typename... Extra>
        This& field(const char* name, D std::remove_pointer_t<T>::* pm, const Extra&... extra)
        {
            this->def_readwrite(name, pm, extra...);

            auto setter = [pm](void* pObj, pybind11::handle h)
            {
                static_cast<T*>(pObj)->*pm = h.cast<D>();
            };

            std::string nameStr(name);
            auto printer = [pm, nameStr](const void* pObj)
            {
                return nameStr + "=" + std::string(pybind11::repr(pybind11::cast(static_cast<const T*>(pObj)->*pm)));
            };

            This::info().fields[name] = { setter, printer };
            return *this;
        }

    private:
        static T init(const pybind11::kwargs& args)
        {
            T t;
            const auto& fields = This::info().fields;
            for (auto a : args) fields.at(a.first.cast<std::string>()).setter(&t, a.second);
            return t;
        }

        static std::string repr(const T& t)
        {
            const auto& info = This::info();
            std::string s = info.name + '(';
            bool first = true;
            for (const auto a : info.fields)
            {
                if (!first) s += ", ";
                first = false;
                s += a.second.printer(&t);
            }
            return s + ')';
        }

        struct Field
        {
            std::function<void(void*, pybind11::handle)> setter;
            std::function<std::string(const void*)> printer;
        };

        struct Info
        {
            std::string name;
            std::unordered_map<std::string, Field> fields;
        };

        static Info& info()
        {
            static Info staticInfo;
            return staticInfo;
        }
    };

#ifndef _staticlibrary
#define SCRIPT_BINDING(Name) \
    static void ScriptBinding##Name(pybind11::module& m);           \
    struct ScriptBindingRegisterer##Name {                          \
        ScriptBindingRegisterer##Name()                             \
        {                                                           \
            ScriptBindings::registerBinding(ScriptBinding##Name);   \
        }                                                           \
    } gScriptBinding##Name;                                         \
    static void ScriptBinding##Name(pybind11::module& m) /* over to the user for the braces */
#else
#define SCRIPT_BINDING(Name) static_assert(false, "Using SCRIPT_BINDING() in a static-library is not supported. The C++ linker usually doesn't pull static-initializers into the EXE. " \
    "Call 'registerBinding()' yourself from a code that is guarenteed to run.");

#endif // _library

}
