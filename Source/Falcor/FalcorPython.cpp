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
#include "Core/API/Device.h"
#include "Core/Plugin.h"
#include "Utils/Scripting/ScriptBindings.h"

#include <pybind11/pybind11.h>

/**
 * This function checks if the falcor module is loaded from a Falcor
 * application running the embedded python interpreter (e.g. Mogwai).
 */
static bool isLoadedFromEmbeddedPython()
{
    try
    {
        auto os = pybind11::module::import("os");
        std::string value = os.attr("environ")["FALCOR_EMBEDDED_PYTHON"].cast<pybind11::str>();
        return value == "1";
    }
    catch (const std::exception&)
    {}

    return false;
}

PYBIND11_MODULE(falcor_ext, m)
{
    if (!isLoadedFromEmbeddedPython())
    {
        Falcor::Logger::setOutputs(Falcor::Logger::OutputFlags::Console | Falcor::Logger::OutputFlags::DebugWindow);
        Falcor::Device::enableAgilitySDK();
        Falcor::PluginManager::instance().loadAllPlugins();
    }

    m.doc() = "Falcor python bindings";
    Falcor::ScriptBindings::initModule(m);
}
