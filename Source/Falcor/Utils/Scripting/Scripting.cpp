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
 **************************************************************************/
#include "stdafx.h"
#include "Scripting.h"
#include <filesystem>
#include "pybind11/embed.h"

namespace Falcor
{
    const FileDialogFilterVec Scripting::kFileExtensionFilters = { { "py", "Script Files"} };
    bool Scripting::sRunning = false;

    template<typename CppType>
    static bool insertNewValue(const std::pair<pybind11::handle, pybind11::handle>& pyVar, Dictionary& falcorDict)
    {
        try
        {
            CppType cppVal = pyVar.second.cast<CppType>();
            std::string name = pyVar.first.cast<std::string>();
            falcorDict[name] = cppVal;
        }
        catch (const std::runtime_error&)
        {
            return false;
        }
        return true;
    }

    static bool insertNewFloatVec(const std::pair<pybind11::handle, pybind11::handle>& pyVar, Dictionary& falcorDict)
    {
        try
        {
            std::vector<float> floatVec = pyVar.second.cast<std::vector<float>>();
            std::string name = pyVar.first.cast<std::string>();

            switch (floatVec.size())
            {
            case 1:
                falcorDict[name] = floatVec[0]; break;
            case 2:
                falcorDict[name] = vec2(floatVec[0], floatVec[1]); break;
            case 3:
                falcorDict[name] = vec3(floatVec[0], floatVec[1], floatVec[2]); break;
            case 4:
                falcorDict[name] = vec4(floatVec[0], floatVec[1], floatVec[2], floatVec[3]); break;
            default:
                falcorDict[name] = floatVec;
            }
        }
        catch (const std::runtime_error&)
        {
            return false;
        }
        return true;
    }

    bool Scripting::start()
    {
        if (!sRunning)
        {
            sRunning = true;
#ifdef _WIN32
            static std::wstring pythonHome = string_2_wstring(getExecutableDirectory() + "/Python");
            Py_SetPythonHome(pythonHome.c_str());
#endif

            try
            {
                pybind11::initialize_interpreter();
                pybind11::exec("from falcor import *");
            }
            catch (const std::exception& e)
            {
                logError("Can't start the python interpreter. Exception says " + std::string(e.what()));
                return false;
            }
        }

        return true;
    }

    void Scripting::shutdown()
    {
        if (sRunning)
        {
            sRunning = false;
            pybind11::finalize_interpreter();
            ScriptBindings::sClasses.clear();
        }
    }

    class RedirectStdout
    {
    public:
        RedirectStdout()
        {
            auto m = pybind11::module::import("sys");
            mOrigOut = m.attr("stdout");
            mBuffer = pybind11::module::import("io").attr("StringIO")();
            m.attr("stdout") = mBuffer;
        }

        ~RedirectStdout()
        {
            pybind11::module::import("sys").attr("stdout") = mOrigOut;
        }

        operator std::string() const
        {
            mBuffer.attr("seek")(0);
            return pybind11::str(mBuffer.attr("read")());
        }

    private:
        pybind11::object mOrigOut;
        pybind11::object mBuffer;
    };

    static std::string runScript(const std::string& script, pybind11::dict& locals)
    {
        RedirectStdout rs;
        pybind11::exec(script.c_str(), pybind11::globals(), locals);
        return rs;
    }

    std::string Scripting::runScript(const std::string& script)
    {
        auto ref = pybind11::globals();
        return Falcor::runScript(script, ref);
    }

    std::string Scripting::runScript(const std::string& script, Context& context)
    {
        return Falcor::runScript(script, context.mLocals);
    }

    Scripting::Context Scripting::getGlobalContext()
    {
        Context c;
        c.mLocals = pybind11::globals();
        return c;
    }

    std::string Scripting::runScriptFromFile(const std::string& filename, Context& context)
    {
        if (std::filesystem::exists(filename)) return Scripting::runScript(readFile(filename), context);
        throw std::exception(std::string("Failed to run script. Can't find the file `" + filename + "`.").c_str());
    }
}
