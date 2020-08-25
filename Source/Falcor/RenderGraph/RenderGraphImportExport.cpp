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
#include "stdafx.h"
#include "RenderGraphImportExport.h"
#include "RenderPassLibrary.h"
#include "RenderGraphIR.h"
#include <fstream>

namespace Falcor
{
    namespace
    {
        void updateGraphStrings(std::string& graph, std::string& file, std::string& func)
        {
            graph = graph.empty() ? "renderGraph" : graph;
            file = file.empty() ? graph + ".py" : file;
            func = func.empty() ? RenderGraphIR::getFuncName(graph) : func;
        }

        void runScriptFile(const std::string& filename, const std::string& custom)
        {
            std::string fullpath;
            if (findFileInDataDirectories(filename, fullpath) == false)
            {
                throw std::exception("Can't find the file");
            }

            std::string script = readFile(fullpath) + custom;
            Scripting::runScript(script);
        }
    }

    bool loadFailed(std::exception e, const std::string& filename)
    {
        logError(e.what(), Logger::MsgBox::None);
        auto res = msgBox(std::string("Error when importing graph from file '" + filename + "'\n" + e.what() + "\n\nWould you like to try and reload the file?").c_str(), MsgBoxType::YesNo);
        return (res == MsgBoxButton::No);
    }

    RenderGraph::SharedPtr RenderGraphImporter::import(std::string graphName, std::string filename, std::string funcName)
    {
        while(true)
        {
            try
            {
                updateGraphStrings(graphName, filename, funcName);
                std::string custom;
                if (funcName.size()) custom += "\n" + graphName + '=' + funcName + "()";
                runScriptFile(filename, custom);

                auto pGraph = Scripting::getGlobalContext().getObject<RenderGraph::SharedPtr>(graphName);
                if (!pGraph) throw("Unspecified error");

                pGraph->setName(graphName);
                return pGraph;
            }
            catch (const std::exception& e)
            {
                if (loadFailed(e, filename)) return nullptr;
            }
        }
    }

    std::vector<RenderGraph::SharedPtr> RenderGraphImporter::importAllGraphs(const std::string& filename)
    {
        while(true)
        {
            try
            {
                runScriptFile(filename, {});
                auto scriptObj = Scripting::getGlobalContext().getObjects<RenderGraph::SharedPtr>();
                std::vector<RenderGraph::SharedPtr> res;
                res.reserve(scriptObj.size());

                for (const auto& s : scriptObj)
                {
                    s.obj->setName(s.name);
                    res.push_back(s.obj);
                }

                return res;
            }
            catch (const std::exception& e)
            {
                if (loadFailed(e, filename)) return {};
            }
        }
    }

    std::string RenderGraphExporter::getFuncName(const std::string& graphName)
    {
        return RenderGraphIR::getFuncName(graphName);
    }

    std::string RenderGraphExporter::getIR(const RenderGraph::SharedPtr& pGraph)
    {
        RenderGraphIR::SharedPtr pIR = RenderGraphIR::create(pGraph->getName());

        // Register passes that are loaded from dlls
        auto libNames = RenderPassLibrary::enumerateLibraries();
        for (const auto& libName : libNames)
        {
            pIR->loadPassLibrary(getFilenameFromPath(libName));
        }

        // Add the passes
        for (const auto& node : pGraph->mNodeData)
        {
            const auto& data = node.second;
            pIR->addPass(getClassTypeName(data.pPass.get()), data.name, data.pPass->getScriptingDictionary());
        }

        // Add the edges
        for (const auto& edge : pGraph->mEdgeData)
        {
            const auto& data = edge.second;
            const auto& srcPass = pGraph->mNodeData[pGraph->mpGraph->getEdge(edge.first)->getSourceNode()].name;
            const auto& dstPass = pGraph->mNodeData[pGraph->mpGraph->getEdge(edge.first)->getDestNode()].name;
            std::string src = srcPass + (data.srcField.size() ? '.' + data.srcField : data.srcField);
            std::string dst = dstPass + (data.dstField.size() ? '.' + data.dstField : data.dstField);
            pIR->addEdge(src, dst);
        }

        // Graph outputs
        for (const auto& out : pGraph->mOutputs)
        {
            std::string str = pGraph->mNodeData[out.nodeId].name + '.' + out.field;
            pIR->markOutput(str);
        }

        return pIR->getIR();
    }

    bool RenderGraphExporter::save(const std::shared_ptr<RenderGraph>& pGraph, std::string filename)
    {
        std::string ir = getIR(pGraph);
        std::string funcName;
        std::string graphName = pGraph->getName();
        updateGraphStrings(graphName, filename, funcName);

        // Save it to file
        std::ofstream f(filename);
        f << ir << std::endl;
        f << graphName << " = " << funcName + "()\n";
        // Try adding it to Mogwai
        f << "try: m.addGraph(" + graphName + ")\n";
        f << "except NameError: None\n";

        return true;
    }
}
