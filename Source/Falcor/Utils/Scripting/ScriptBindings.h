/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "pybind11/stl.h"

namespace Falcor::ScriptBindings
{
    struct enable_to_string {};
    class Module;

    // Helper to check if a class has a `SharedPtr`. If it is, we're using it as the internal python object
    template<typename T, typename = void>
    struct has_shared_ptr : std::false_type {};

    template<typename T>
    struct has_shared_ptr<T, std::void_t<typename T::SharedPtr>> : std::true_type {};

    /************************************************************************/
    /* Namespace definitions                                                */
    /************************************************************************/
    struct ClassDesc
    {
        ClassDesc() = default;
        ClassDesc(const std::string& n) : name(n) {};
        struct Funcs
        {
            std::function<void(void*, pybind11::handle)> setF;
            std::function<std::string(const void*)> printF;
        };
        std::unordered_map<std::string, Funcs> funcs;
        std::string name;
    };

    using ClassesMap = std::unordered_map<std::type_index, ClassDesc>;
    dlldecl extern ClassesMap sClasses;
    dlldecl extern std::unordered_map<std::type_index, std::string> sEnumNames;

    using BindComponentFunc = std::function<void(ScriptBindings::Module& m)>;
    dlldecl void registerBinding(BindComponentFunc f);

    /** A custom to_string() that handles registered classes
    */
    using Falcor::to_string;

    template<typename T>
    typename std::enable_if_t<std::is_base_of_v<enable_to_string, T>, std::string> to_string(const T& t)
    {
        assert(sClasses.find(typeid(T)) != sClasses.end());
        std::string s = sClasses.at(typeid(T)).name + '(';
        bool first = true;
        for (const auto a : sClasses.at(typeid(T)).funcs)
        {
            if (!first) s += ", ";
            first = false;
            s += a.second.printF(&t);
        }
        return s + ")";
    }

    /************************************************************************/
    /* Class                                                                */
    /************************************************************************/
    template<typename T, typename... Options>
    class Class
    {
    public:
        template<typename D, typename... Extra>
        Class& rwField(const char* name, D std::remove_pointer_t<T>::* pm, const Extra&... extra)
        {
            auto setF = [pm](void* pObj, pybind11::handle h) { static_cast<T*>(pObj)->*pm = h.cast<D>(); };
            std::string nameStr(name);
            auto printF = [pm, nameStr](const void* pObj)
            {
                auto s = nameStr + "=";
                if constexpr(std::is_enum_v<D>) s += sEnumNames.at(typeid(D)) + ".";
                s += to_string(static_cast<const T*>(pObj)->*pm);
                return s;
            };

            sClasses[typeid(T)].funcs[name] = { setF, printF };
            pyclass.def_readwrite(name, pm, extra...);
            return *this;
        }

        template <typename Func, typename... Extra>
        Class& func_(const char* name, Func&& f, const Extra&... extra)
        {
            pyclass.def(name, std::forward<Func>(f), extra...);
            return *this;
        }

        template <typename Func, typename... Extra>
        Class& ctor(Func&& f, const Extra&... extra)
        {
            pyclass.def(pybind11::init(f), extra...);
            return *this;
        }

        template <typename Func, typename... Extra>
        Class& staticFunc_(const char* name, Func&& f, const Extra&... extra)
        {
            pyclass.def_static(name, std::forward<Func>(f), extra...);
            return *this;
        }
    private:
        friend Module;

        Class(const char* name, pybind11::module& m) : pyclass(m, name)
        {
            if constexpr(std::is_default_constructible_v<T> && std::is_copy_constructible_v<T>)
            {
                sClasses[typeid(T)] = ClassDesc(name);
                auto initFunc = [](const pybind11::kwargs& args)
                {
                    T t;
                    const auto& classBindings = sClasses.at(typeid(T)).funcs;
                    for (auto a : args) classBindings.at(a.first.cast<std::string>()).setF(&t, a.second);
                    return t;
                };
                pyclass.def(pybind11::init(initFunc)).def(pybind11::init<>());
                if constexpr(std::is_base_of_v<enable_to_string, T>) pyclass.def("__repr__", to_string<T>);
            }
        }
        pybind11::class_<T, Options...> pyclass;
    };

    /************************************************************************/
    /* Enum                                                                 */
    /************************************************************************/
    template<typename T>
    class Enum
    {
    public:
        Enum& value(const char* name, T value)
        {
            pyenum.value(name, value);
            return *this;
        }
    private:
        friend Module;
        Enum(const char* name, pybind11::module& m) : pyenum(m, name) { sEnumNames[typeid(T)] = name; }
        pybind11::enum_<T> pyenum;
    };

    /************************************************************************/
    /* Module                                                               */
    /************************************************************************/
    class Module
    {
    public:
        // An overload of class_ which will be invoked if the object has SharedPtr
        template<typename T, typename... Options>
        auto class_(const char* name)
        {
            if (classExists<T>())
            {
                throw std::runtime_error((std::string("Class ") + name + " was already registered").c_str());
            }

            if constexpr(has_shared_ptr<T>::value)
            {
                return Class<T, T::SharedPtr, Options...>(name, mModule);
            }
            else
            {
                return Class<T, Options...>(name, mModule);
            }
        }

        template<typename T>
        Enum<T> enum_(const char* name)
        {
            return Enum<T>(name, mModule);
        }

        template <typename Func, typename... Extra>
        Module& func_(const char* name, Func&& f, const Extra&... extra)
        {
            mModule.def(name, std::forward<Func>(f), extra...);
            return *this;
        }

        Module(pybind11::module& m) : mModule(m) {}

        template<typename T>
        bool classExists() const
        {
            try
            {
                pybind11::dict d;
                d["test"] = (T*)nullptr;
                return true;
            }
            catch (std::exception) { return false; }
        }
    private:
        pybind11::module& mModule;
    };

    using pybind11::overload_cast;
    using pybind11::const_;

    /************************************************************************/
    /* Helpers                                                              */
    /************************************************************************/

#ifndef _staticlibrary
#define SCRIPT_BINDING(Name) \
    static void ScriptBinding##Name(ScriptBindings::Module& m);       \
    struct ScriptBindingRegisterer##Name {                            \
        ScriptBindingRegisterer##Name()                               \
        {                                                             \
            ScriptBindings::registerBinding(ScriptBinding##Name);     \
        }                                                             \
    } gScriptBinding##Name;                                           \
    static void ScriptBinding##Name(ScriptBindings::Module& m) /* over to the user for the braces */
#else
#define SCRIPT_BINDING(Name) static_assert(false, "Using SCRIPT_BINDING() in a static-library is not supported. The C++ linker usually doesn't pull static-initializers into the EXE. " \
    "Call `registerBinding()` yourself from a code that is guarenteed to run.");

#endif // _library

#define regEnumVal(a) value(to_string(a).c_str(), a)
#define regClass(c_) class_<c_>(#c_);
}
