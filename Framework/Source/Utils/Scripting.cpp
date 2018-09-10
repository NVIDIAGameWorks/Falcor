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
#include "Framework.h"
#include "Scripting.h"
#include "Externals/pybind11-2.2.3/include/pybind11/embed.h"
#include "Externals/pybind11-2.2.3/include/pybind11/stl.h"
#include "StringUtils.h"
#include "Utils/Dictionary.h"

#include "RenderGraphScripting.h"

using namespace pybind11::literals;

namespace Falcor
{
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

    static void convertDouble(const std::string& name, Dictionary& falcorDict)
    {
        double d = falcorDict[name].asDouble();

        // Order matters
        if (fract(d) == 0)
        {
            // UINT first. Double fits into uint64_t, so no need to check for anything
            if(d >= 0)
            {
                if (d <= UINT32_MAX) falcorDict[name] = uint32_t(d);
                else falcorDict[name] = uint64_t(d);
            }
            // INTs. Double has larger range, so make sure we can fit
            else if(d >= INT64_MIN)
            {
                if (d >= INT32_MIN) falcorDict[name] = int32_t(d);
                else falcorDict[name] = int64_t(d);
            }
        }
        else if (d <= FLT_MAX && d >= -FLT_MAX)
        {
            falcorDict[name] = float(d);
        }
    }

    Dictionary convertPythonDict(const pybind11::dict& pyDict)
    {
        Dictionary falcorDict;
        for (const auto& d : pyDict)
        {
            // The order matters here, since pybind11 does implicit conversion if it can
            if (insertNewValue<double>(d, falcorDict))
            {
                convertDouble(d.first.cast<std::string>(), falcorDict);
                continue;
            }
            if (insertNewValue<std::string>(d, falcorDict)) continue;
            if (insertNewFloatVec(d, falcorDict)) continue;
            should_not_get_here();
        }

        return falcorDict;
    }

    PYBIND11_EMBEDDED_MODULE(falcor, m)
    {
        RenderGraphScripting::registerScriptingObjects(m);
    }

    bool Scripting::start()
    {
        if (!sRunning)
        {
            sRunning = true;
            static const std::wstring pythonHome = string_2_wstring(std::string(_PROJECT_DIR_) + "/../Externals/Python37");
            Py_SetPythonHome(const_cast<wchar_t*>(pythonHome.c_str()));

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
        }
    }

    static bool runScript(const std::string& script, std::string& errorLog, pybind11::dict& locals)
    {
        try
        {
            pybind11::exec(script.c_str(), pybind11::globals(), locals);
        }
        catch (const std::runtime_error& e)
        {
            errorLog = e.what();
            return false;
        }

        return true;
    }

    bool Scripting::runScript(const std::string& script, std::string& errorLog)
    {
        return Falcor::runScript(script, errorLog, pybind11::globals());
    }

    bool Scripting::runScript(const std::string& script, std::string& errorLog, Context& context)
    {
        return Falcor::runScript(script, errorLog, context.mLocals);
    }

    Scripting::Context Scripting::getGlobalContext() const
    {
        Context c;
        c.mLocals = pybind11::globals();
        return c;
    }
}