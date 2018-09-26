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
#include "RenderGraphImportExport.h"
#include "RenderGraphScripting.h"
#include "RenderGraphIR.h"
#include <fstream>

namespace Falcor
{
    static void updateGraphStrings(const std::string& graph, std::string& file, std::string& func)
    {
        file = file.empty() ? graph + ".py" : file;
        func = func.empty() ? "render_graph_" + graph : func;
    }

    RenderGraph::SharedPtr RenderGraphImporter::import(std::string graphName, std::string filename, std::string funcName, const Fbo* pDstFbo)
    {
        bool gotFuncName = funcName.size();
        updateGraphStrings(graphName, filename, funcName);

        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logError("Error when loading graph. Can't find the file `" + filename + "`");
            return nullptr;
        }

        // Run the script and try to get the graph
        RenderGraphScripting::SharedPtr pScripting = RenderGraphScripting::create(fullpath);
        RenderGraph::SharedPtr pGraph;
        if(gotFuncName) pGraph = pScripting->getGraph(graphName);

        if(pGraph == nullptr)
        {
            // If we didn't succeed or got a custom function name, try and call the graph function explicitly
            if (pScripting->runScript(graphName + '=' + funcName + "()") == false) return nullptr;
            pGraph = pScripting->getGraph(graphName);
        }

        if (pGraph && pDstFbo) pGraph->onResize(pDstFbo);

        return pGraph;
    }

    std::vector<RenderGraphImporter::GraphData> RenderGraphImporter::importAllGraphs(const std::string& filename, const Fbo* pDstFbo)
    {
        RenderGraphScripting::SharedPtr pScripting = RenderGraphScripting::create(filename);
        if (!pScripting) return {};

        const auto& scriptVec = pScripting->getGraphs();
        std::vector<RenderGraphImporter::GraphData> res;
        res.reserve(scriptVec.size());

        for (const auto& s : scriptVec)
        {
            if(pDstFbo) s.obj->onResize(pDstFbo);
            res.push_back({ s.name, s.obj });
        }

        return res;
    }

    bool RenderGraphExporter::save(const std::shared_ptr<RenderGraph>& pGraph, std::string graphName, std::string filename, std::string funcName)
    {
        updateGraphStrings(graphName, filename, funcName);
        RenderGraphIR::SharedPtr pIR = RenderGraphIR::create(graphName);

        // Add the passes
        for (const auto& node : pGraph->mNodeData)
        {
            const auto& data = node.second;
            pIR->addPass(data.pPass->getName(), data.nodeName, data.pPass->getScriptingDictionary());
        }

        // Add the edges
        for (const auto& edge : pGraph->mEdgeData)
        {
            const auto& data = edge.second;
            const auto& srcPass = pGraph->mNodeData[pGraph->mpGraph->getEdge(edge.first)->getSourceNode()].nodeName;
            const auto& dstPass = pGraph->mNodeData[pGraph->mpGraph->getEdge(edge.first)->getDestNode()].nodeName;
            std::string src = srcPass + '.' + data.srcField;
            std::string dst = dstPass + '.' + data.dstField;
            pIR->addEdge(src, dst);
        }

        // Graph outputs
        for (const auto& out : pGraph->mOutputs)
        {
            std::string str = pGraph->mNodeData[out.nodeId].nodeName + '.' + out.field;
            pIR->markOutput(str);
        }

        // Save it to file
        std::ofstream f(filename);
        f << pIR->getIR() << std::endl;
        f << graphName << " = " << funcName + "()";
        return true;
    }
}