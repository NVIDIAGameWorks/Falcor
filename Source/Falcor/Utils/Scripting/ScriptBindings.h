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
#include "Core/Error.h"
#include "Core/Enum.h"
#include "Core/ObjectPython.h"
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <functional>
#include <string>
#include <type_traits>

namespace Falcor::ScriptBindings
{
/**
 * Callback function to add python bindings to a module.
 */
using RegisterBindingFunc = std::function<void(pybind11::module& m)>;

/**
 * Initialize the binding module.
 * This is called from the pybind11 module to initialize bindings (see FalcorPython.cpp).
 * First, this function will go through the list of all deferred bindings and execute them.
 * Next, it stores a reference to the python module to allow further bindings to be added at runtime.
 * The reference to the module is automatically released when the module is unloaded.
 */
FALCOR_API void initModule(pybind11::module& m);

/**
 * Register a script binding function.
 * The binding function will be called when scripting is initialized.
 * @param[in] f Function to be called for registering the binding.
 */
FALCOR_API void registerBinding(RegisterBindingFunc f);

/**
 * Register a deferred script binding function.
 * This is used to register a script binding function before scripting is initialized.
 * The execution of the binding function is deferred until scripting is finally initialized.
 * Note: This is called from `registerBinding()` if called before scripting is initialized
 * and from the FALCOR_SCRIPT_BINDING macro.
 * @param[in] name Name if the binding.
 * @param[in] f Function to be called for registering the binding.
 */
FALCOR_API void registerDeferredBinding(const std::string& name, RegisterBindingFunc f);

/**
 * Resolve a deferred script binding by name.
 * This immediately executes the deferred binding function registered to the given name
 * and can be used to control the order of execution of the binding functions.
 * Note: This is used by the FALCOR_SCRIPT_BINDING_DEPENDENCY macro to ensure dependent bindings
 * are registered ahead of time.
 * @param[in] name Name of the binding to resolve.
 * @param[in] m Python module.
 */
FALCOR_API void resolveDeferredBinding(const std::string& name, pybind11::module& m);

/************************************************************************/
/* Helpers                                                              */
/************************************************************************/

/**
 * Adds binary and/or operators to a Python enum.
 * This allows the enum to be used as a set of flags instead of just a list of choices.
 * @param[in] e Enum to be extended.
 */
template<typename T>
static void addEnumBinaryOperators(pybind11::enum_<T>& e)
{
    e.def("__and__", [](const T& value1, const T& value2) { return T(int(value1) & int(value2)); });
    e.def("__or__", [](const T& value1, const T& value2) { return T(int(value1) | int(value2)); });
}

/**
 * Returns the string representation of a value of a registered type.
 * @param[in] value Value to be converted to a string.
 * @return Returns the string representation.
 */
template<typename T>
static std::string repr(const T& value)
{
    return pybind11::repr(pybind11::cast(value));
}

#ifndef _staticlibrary
#define FALCOR_SCRIPT_BINDING(_name)                                               \
    static void ScriptBinding##_name(pybind11::module& m);                         \
    struct ScriptBindingRegisterer##_name                                          \
    {                                                                              \
        ScriptBindingRegisterer##_name()                                           \
        {                                                                          \
            ScriptBindings::registerDeferredBinding(#_name, ScriptBinding##_name); \
        }                                                                          \
    } gScriptBinding##_name;                                                       \
    static void ScriptBinding##_name(pybind11::module& m) /* over to the user for the braces */
#define FALCOR_SCRIPT_BINDING_DEPENDENCY(_name) ScriptBindings::resolveDeferredBinding(#_name, m);
#else
#define FALCOR_SCRIPT_BINDING(_name)                                                                                                   \
    static_assert(                                                                                                                     \
        false,                                                                                                                         \
        "Using FALCOR_SCRIPT_BINDING() in a static-library is not supported. The C++ linker usually doesn't pull static-initializers " \
        "into the EXE. "                                                                                                               \
        "Call 'registerBinding()' yourself from a code that is guarenteed to run."                                                     \
    );
#endif // _staticlibrary

} // namespace Falcor::ScriptBindings

namespace pybind11
{

template<typename T>
class falcor_enum : public enum_<T>
{
public:
    static_assert(::Falcor::has_enum_info_v<T>, "pybind11::falcor_enum<> requires an enumeration type with infos!");

    using Base = enum_<T>;

    template<typename... Extra>
    explicit falcor_enum(const handle& scope, const char* name, const Extra&... extra) : Base(scope, name, extra...)
    {
        for (const auto& item : ::Falcor::EnumInfo<T>::items())
        {
            const char* value_name = item.second.c_str();
            // Handle reserved Python keywords.
            if (item.second == "None")
                value_name = "None_";
            Base::value(value_name, item.first);
        }
    }
};
} // namespace pybind11
