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
#include "Core/Macros.h"
#include "Core/Platform/OS.h"
#include <pybind11/pybind11.h>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace Falcor
{
    class FALCOR_API Scripting
    {
    public:
        static const FileDialogFilterVec kFileExtensionFilters;

        /** Represents a context for executing scripts.
            Wraps the globals dictionary that is passed to the script on execution.
            The context can be used to pass/retrieve variables to/from the executing script.
        */
        class Context
        {
        public:
            Context(pybind11::dict globals) : mGlobals(globals) {}

            Context()
            {
                // Copy __builtins__ to our empty globals dictionary.
                mGlobals["__builtins__"] = pybind11::globals()["__builtins__"];
            }

            template<typename T>
            struct ObjectDesc
            {
                ObjectDesc(const std::string& name_, const T& obj_) : name(name_), obj(obj_) {}
                operator const T&() const { return obj; }
                std::string name;
                T obj;
            };

            template<typename T>
            std::vector<ObjectDesc<T>> getObjects()
            {
                std::vector<ObjectDesc<T>> v;
                for (const auto& l : mGlobals)
                {
                    try
                    {
                        if(!l.second.is_none())
                        {
                            v.push_back(ObjectDesc<T>(l.first.cast<std::string>(), l.second.cast<T>()));
                        }
                    }
                    catch (const std::exception&) {}
                }
                return v;
            }

            template<typename T>
            void setObject(const std::string& name, T obj)
            {
                mGlobals[name.c_str()] = obj;
            }

            template<typename T>
            T getObject(const std::string& name) const
            {
                return mGlobals[name.c_str()].cast<T>();
            }

            bool containsObject(const std::string& name) const
            {
                return mGlobals.contains(name.c_str());
            }

        private:
            friend class Scripting;
            pybind11::dict mGlobals;
        };

        /** Starts the script engine.
            This will initialize the Python interpreter and setup the default context.
        */
        static void start();

        /** Shuts the script engine down.
        */
        static void shutdown();

        /** Returns true if the script engine is running.
        */
        static bool isRunning() { return sRunning; }

        /** Returns the default context.
        */
        static Context& getDefaultContext();

        /** Returns the context of the currently executing script.
        */
        static Context getCurrentContext();

        struct RunResult
        {
            std::string out;
            std::string err;
        };

        /** Run a script.
            \param[in] script Script to run.
            \param[in] context Script execution context.
            \param[in] captureOutput Enable capturing stdout/stderr and returning it in RunResult.
            \return Returns the captured output if enabled.
        */
        static RunResult runScript(const std::string& script, Context& context = getDefaultContext(), bool captureOutput = false);

        /** Run a script from a file.
            \param[in] path Path of the script to run.
            \param[in] context Script execution context.
            \param[in] captureOutput Enable capturing stdout/stderr and returning it in RunResult.
            \return Returns the captured output if enabled.
        */
        static RunResult runScriptFromFile(const std::filesystem::path& path, Context& context = getDefaultContext(), bool captureOutput = false);

        /** Interpret a script and return the evaluated result.
            \param[in] script Script to run.
            \param[in] context Script execution context.
            \return Returns a string representation of the evaluated result of the script.
        */
        static std::string interpretScript(const std::string& script, Context& context = getDefaultContext());

    private:
        static bool sRunning;
        static std::unique_ptr<Context> sDefaultContext;
    };
}
