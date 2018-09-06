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
#include "Externals/pybind11-2.2.3/include/pybind11/pybind11.h"
#include "Externals/pybind11-2.2.3/include/pybind11/embed.h"
#include "StringUtils.h"

#include "Graphics/RenderGraph/RenderGraph.h"
#include "Graphics/RenderGraph/RenderPassesLibrary.h"

using namespace pybind11;

namespace Falcor
{
    bool Scripting::sRunning = false;

    PYBIND11_EMBEDDED_MODULE(falcor, m)
    {
        m.def("createRenderGraph", &RenderGraph::create);
        m.def("createRenderPass", &RenderPassLibrary::createRenderPass);

        void(RenderGraph::*renderGraphRemoveEdge)(const std::string&, const std::string&)(&RenderGraph::removeEdge);
        pybind11::class_<RenderGraph, RenderGraph::SharedPtr>(m, "Graph").def("addRenderPass", &RenderGraph::addRenderPass).def("addEdge", &RenderGraph::addEdge).def("removeRenderPass", &RenderGraph::removeRenderPass).def("removeEdge", renderGraphRemoveEdge);

        pybind11::class_<RenderPass, RenderPass::SharedPtr>(m, "RenderPass");
    }

    bool Scripting::start()
    {
        if (!sRunning)
        {
            sRunning = true;
            static const std::wstring pythonHome = string_2_wstring(std::string(_PROJECT_DIR_) + "/../Externals/Python37");
            Py_SetPythonHome(pythonHome.c_str());

            try
            {
                initialize_interpreter();
                exec("from falcor import *");
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
            finalize_interpreter();
        }
    }

    bool Scripting::runScript(const std::string& script, std::string& errorLog)
    {
        try
        {
            exec(script.c_str());
        }
        catch (const std::runtime_error& e)
        {
            errorLog = e.what();
            return false;
        }

        return true;
    }
}