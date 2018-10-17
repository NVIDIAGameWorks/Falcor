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
#include "RenderGraphScripting.h"
#include "Utils/Scripting/Scripting.h"
#include <fstream>
#include <sstream>
#include "Graphics/RenderGraph/RenderPassLibrary.h"
#include "pybind11/operators.h"
#include "Graphics/Scene/Scene.h"

using namespace pybind11::literals;

namespace Falcor
{
    const char* RenderGraphScripting::kAddPass = "addPass";
    const char* RenderGraphScripting::kRemovePass = "removePass";
    const char* RenderGraphScripting::kAddEdge = "addEdge";
    const char* RenderGraphScripting::kRemoveEdge = "removeEdge";
    const char* RenderGraphScripting::kMarkOutput = "markOutput";
    const char* RenderGraphScripting::kUnmarkOutput = "unmarkOutput";
    const char* RenderGraphScripting::kAutoGenEdges = "autoGenEdges";
    const char* RenderGraphScripting::kCreateGraph = "createRenderGraph";
    const char* RenderGraphScripting::kCreatePass = "createRenderPass";
    const char* RenderGraphScripting::kUpdatePass = "updatePass";
    const char* RenderGraphScripting::kLoadPassLibrary = "loadRenderPassLibrary";
    const char* RenderGraphScripting::kSetName = "setName";
    const char* RenderGraphScripting::kSetScene = "setScene";

    void RenderGraphScripting::registerScriptingObjects(pybind11::module& m)
    {
        // RenderGraph
        m.def(kCreateGraph, &RenderGraph::create, "name"_a = "");

        void(RenderGraph::*renderGraphRemoveEdge)(const std::string&, const std::string&)(&RenderGraph::removeEdge);
        auto graphClass = pybind11::class_<RenderGraph, RenderGraph::SharedPtr>(m, "Graph");
        graphClass.def(kAddPass, &RenderGraph::addPass).def(kRemovePass, &RenderGraph::removePass);
        graphClass.def(kAddEdge, &RenderGraph::addEdge).def(kRemoveEdge, renderGraphRemoveEdge);
        graphClass.def(kMarkOutput, &RenderGraph::markOutput).def(kUnmarkOutput, &RenderGraph::unmarkOutput);
        graphClass.def(kAutoGenEdges, &RenderGraph::autoGenEdges);
        graphClass.def(kSetName, &RenderGraph::setName);
        graphClass.def(kSetScene, &RenderGraph::setScene);

        // RenderPass
        pybind11::class_<RenderPass, RenderPass::SharedPtr>(m, "RenderPass");

        // RenderPassLibrary
        const auto& createRenderPass = [](const std::string& passName, pybind11::dict d = {})->RenderPass::SharedPtr
        {
            return RenderPassLibrary::instance().createPass(passName.c_str(), Dictionary(d));
        };
        m.def(kCreatePass, createRenderPass, "passName"_a, "dict"_a = pybind11::dict());

        const auto& loadPassLibrary = [](const std::string& library)
        {
            return RenderPassLibrary::instance().loadLibrary(library);
        };
        m.def(kLoadPassLibrary, loadPassLibrary);

        const auto& updateRenderPass = [](const RenderGraph::SharedPtr& pGraph, const std::string& passName, pybind11::dict d )
        {
            pGraph->updatePass(passName, Dictionary(d));
        };
        graphClass.def(kUpdatePass, updateRenderPass);
    }

    RenderGraphScripting::SharedPtr RenderGraphScripting::create()
    {
        return SharedPtr(new RenderGraphScripting());
    }

    RenderGraphScripting::SharedPtr RenderGraphScripting::create(const std::string& filename)
    {
        SharedPtr pThis = create();

        if (findFileInDataDirectories(filename, pThis->mFilename) == false)
        {
            logError("Error when opening render graphs script file. Can't find the file `" + filename + "`");
            return nullptr;
        }

        pThis->runScript(readFile(pThis->mFilename));
        return pThis;
    }

    bool RenderGraphScripting::runScript(const std::string& script)
    {
        std::string log;
        if (Scripting::runScript(script, log, mContext) == false)
        {
            logError("Can't run render-graphs script.\n" + log);
            return false; 
        }

        mGraphVec = mContext.getObjects<RenderGraph::SharedPtr>();
        return true;
    }

    void RenderGraphScripting::addGraph(const std::string& name, const RenderGraph::SharedPtr& pGraph)
    {
        try
        {
            mContext.getObject<RenderGraph::SharedPtr>(name);
            logWarning("RenderGraph `" + name + "` already exists. Replacing the current object");
        }
        catch (std::exception) {}
        mContext.setObject(name, pGraph);
    }

    RenderGraph::SharedPtr RenderGraphScripting::getGraph(const std::string& name) const
    {
        try
        {
            return mContext.getObject<RenderGraph::SharedPtr>(name);
        }
        catch (std::exception) 
        {
            logWarning("Can't find RenderGraph `" + name + "` in RenderGraphScriptContext");
            return nullptr;
        }
    }
}
