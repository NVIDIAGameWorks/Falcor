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
#include "Scripting.h"
#include "Core/Errors.h"
#include "Utils/StringUtils.h"
#include "Utils/StringFormatters.h"
#include <pybind11/embed.h>
#include <filesystem>

namespace Falcor
{
    const FileDialogFilterVec Scripting::kFileExtensionFilters = { { "py", "Script Files"} };
    bool Scripting::sRunning = false;
    std::unique_ptr<Scripting::Context> Scripting::sDefaultContext;

    void Scripting::start()
    {
        if (!sRunning)
        {
            sRunning = true;
#if FALCOR_WINDOWS
            static std::wstring pythonHome = (getExecutableDirectory() / "Python").c_str();
#else
            static std::wstring pythonHome = string_2_wstring((getExecutableDirectory() / "python").string());
#endif
            // Py_SetPythonHome in Python < 3.7 takes a non-const wstr*, but guarantees that the contents
            // will not be modified by Python. As such, casting away the const should be safe.
            Py_SetPythonHome(const_cast<wchar_t*>(pythonHome.c_str()));

            try
            {
                pybind11::initialize_interpreter();
                sDefaultContext.reset(new Context());
                // Import falcor into default scripting context.
                Scripting::runScript("from falcor import *");
            }
            catch (const std::exception& e)
            {
                throw RuntimeError("Failed to start the Python interpreter: {}", e.what());
            }
        }
    }

    void Scripting::shutdown()
    {
        if (sRunning)
        {
            sRunning = false;
            sDefaultContext.reset();
            pybind11::finalize_interpreter();
        }
    }

    Scripting::Context& Scripting::getDefaultContext()
    {
        FALCOR_ASSERT(sDefaultContext);
        return *sDefaultContext;
    }

    Scripting::Context Scripting::getCurrentContext()
    {
        return Context(pybind11::globals());
    }

    class RedirectStream
    {
    public:
        RedirectStream(const std::string& stream = "stdout")
            : mStream(stream)
        {
            auto m = pybind11::module::import("sys");
            mOrigStream = m.attr(mStream.c_str());
            mBuffer = pybind11::module::import("io").attr("StringIO")();
            m.attr(mStream.c_str()) = mBuffer;
        }

        ~RedirectStream()
        {
            pybind11::module::import("sys").attr(mStream.c_str()) = mOrigStream;
        }

        operator std::string() const
        {
            mBuffer.attr("seek")(0);
            return pybind11::str(mBuffer.attr("read")());
        }

    private:
        std::string mStream;
        pybind11::object mOrigStream;
        pybind11::object mBuffer;
    };

    static Scripting::RunResult runScript(const std::string& script, pybind11::dict& globals, bool captureOutput)
    {
        Scripting::RunResult result;

        if (captureOutput)
        {
            RedirectStream rstdout("stdout");
            RedirectStream rstderr("stderr");
            pybind11::exec(script.c_str(), globals);
            result.out = rstdout;
            result.err = rstderr;
        }
        else
        {
            pybind11::exec(script.c_str(), globals);
        }

        return result;
    }

    Scripting::RunResult Scripting::runScript(const std::string& script, Context& context, bool captureOutput)
    {
        return Falcor::runScript(script, context.mGlobals, captureOutput);
    }

    Scripting::RunResult Scripting::runScriptFromFile(const std::filesystem::path& path, Context& context, bool captureOutput)
    {
        if (std::filesystem::exists(path))
        {
            std::string absFile = std::filesystem::absolute(path).string();
            context.setObject("__file__", absFile);
            auto result = Scripting::runScript(readFile(path), context, captureOutput);
            context.setObject("__file__", nullptr); // There seems to be no API on pybind11::dict to remove a key.
            return result;
        }
        throw RuntimeError("Failed to run script. Can't find the file '{}'.", path);
    }

    std::string Scripting::interpretScript(const std::string& script, Context& context)
    {
        pybind11::module code = pybind11::module::import("code");
        pybind11::object InteractiveInterpreter = code.attr("InteractiveInterpreter");
        auto interpreter = InteractiveInterpreter(context.mGlobals);
        auto runsource = interpreter.attr("runsource");

        RedirectStream rstdout("stdout");
        RedirectStream rstderr("stderr");
        runsource(script);
        return std::string(rstdout) + std::string(rstderr);
    }
}
