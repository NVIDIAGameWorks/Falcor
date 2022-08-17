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
#include "Core/Errors.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // export
#include <pybind11/stl/filesystem.h> // export
#include <functional>
#include <string>
#include <type_traits>

namespace Falcor::ScriptBindings
{
    using RegisterBindingFunc = std::function<void(pybind11::module& m)>;

    /** Register a script binding function.
        The binding function will be called when scripting is initialized.
        \param[in] f Function to be called for registering the binding.
    */
    FALCOR_API void registerBinding(RegisterBindingFunc f);

    /** Register a deferred script binding function.
        This is used to register a script binding function before scripting is initialized.
        The execution of the binding function is deferred until scripting is finally initialized.
        Note: This is called from `registerBinding()` if called before scripting is initialized
        and from the FALCOR_SCRIPT_BINDING macro.
        \param[in] name Name if the binding.
        \param[in] f Function to be called for registering the binding.
    */
    FALCOR_API void registerDeferredBinding(const std::string& name, RegisterBindingFunc f);

    /** Resolve a deferred script binding by name.
        This immediately executes the deferred binding function registered to the given name
        and can be used to control the order of execution of the binding functions.
        Note: This is used by the FALCOR_SCRIPT_BINDING_DEPENDENCY macro to ensure dependent bindings
        are registered ahead of time.
        \param[in] name Name of the binding to resolve.
        \param[in] m Python module.
    */
    FALCOR_API void resolveDeferredBinding(const std::string &name, pybind11::module& m);

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

    /** This helper creates script bindings for simple data-only structs.

        The structs are made "serializable" to/from python code by adding a __init__ (constructor)
        and a __repr__ implementation. The __init__ function takes kwargs and populates all the
        struct fields that have been registered with the field() method on this helper.
        The __repr__ implementation prints all the fields registered with the field() method.

        This helper also makes the python type "pickle-able" by providing a __getstate__
        and __setstate__ implementation.

        Lets assume we have a C++ struct:

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
            this->def(pybind11::pickle(
                [] (const T &obj) { return This::getState(obj); },
                [] (pybind11::tuple t) { T obj; This::setState(obj, t); return obj; }
            ));
        }

        template<typename D, typename... Extra>
        This& field(const char* name, D std::remove_pointer_t<T>::* pm, const Extra&... extra)
        {
            this->def_readwrite(name, pm, extra...);

            auto getter = [pm](const T& obj) -> pybind11::object
            {
                return pybind11::cast(obj.*pm);
            };

            auto setter = [pm](T& obj, pybind11::handle h)
            {
                obj.*pm = h.cast<D>();
            };

            std::string nameStr(name);
            auto printer = [pm, nameStr](const T& obj)
            {
                return nameStr + "=" + std::string(pybind11::repr(pybind11::cast(obj.*pm)));
            };

            auto &info = This::info();
            auto field = Field { getter, setter, printer };
            info.fields.emplace_back(field);
            info.fieldByName[name] = field;
            return *this;
        }

    private:
        static T init(const pybind11::kwargs& args)
        {
            T obj;
            const auto& fieldByName = This::info().fieldByName;
            for (auto a : args) fieldByName.at(a.first.cast<std::string>()).setter(obj, a.second);
            return obj;
        }

        static std::string repr(const T& obj)
        {
            const auto& info = This::info();
            std::string s = info.name + '(';
            bool first = true;
            for (const auto f : info.fields)
            {
                if (!first) s += ", ";
                first = false;
                s += f.printer(obj);
            }
            return s + ')';
        }

        static pybind11::tuple getState(const T &obj)
        {
            const auto& fields = This::info().fields;
            pybind11::tuple t(fields.size());
            for (size_t i = 0; i < fields.size(); ++i)
            {
                t[i] = fields[i].getter(obj);
            }
            return t;
        }

        static void setState(T &obj, pybind11::tuple t)
        {
            const auto& fields = This::info().fields;
            if (t.size() != fields.size()) throw RuntimeError("Invalid state!");
            for (size_t i = 0; i < fields.size(); ++i)
            {
                fields[i].setter(obj, t[i]);
            }
        }

        struct Field
        {
            std::function<pybind11::object(const T&)> getter;
            std::function<void(T&, pybind11::handle)> setter;
            std::function<std::string(const T&)> printer;
        };

        struct Info
        {
            std::string name;
            std::vector<Field> fields;
            std::unordered_map<std::string, Field> fieldByName;
        };

        static Info& info()
        {
            static Info staticInfo;
            return staticInfo;
        }
    };

#ifndef _staticlibrary
#define FALCOR_SCRIPT_BINDING(_name)                                                \
    static void ScriptBinding##_name(pybind11::module& m);                          \
    struct ScriptBindingRegisterer##_name {                                         \
        ScriptBindingRegisterer##_name()                                            \
        {                                                                           \
            ScriptBindings::registerDeferredBinding(#_name, ScriptBinding##_name);  \
        }                                                                           \
    } gScriptBinding##_name;                                                        \
    static void ScriptBinding##_name(pybind11::module& m) /* over to the user for the braces */
#define FALCOR_SCRIPT_BINDING_DEPENDENCY(_name)                                     \
    ScriptBindings::resolveDeferredBinding(#_name, m);
#else
#define FALCOR_SCRIPT_BINDING(_name) static_assert(false, "Using FALCOR_SCRIPT_BINDING() in a static-library is not supported. The C++ linker usually doesn't pull static-initializers into the EXE. " \
    "Call 'registerBinding()' yourself from a code that is guarenteed to run.");

#endif // _library

}
