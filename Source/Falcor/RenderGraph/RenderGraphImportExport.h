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
#include "RenderGraph.h"
#include "Core/Macros.h"
#include <filesystem>
#include <string>
#include <vector>

namespace Falcor
{
    class FALCOR_API RenderGraphImporter
    {
    public:
        /** Import a graph from a file.
            \param[in] graphName The name of the graph to import
            \param[in] path The graphs file path. If the path is empty, the function will search for a file called `<graphName>.py`
            \param[in] funcName The function name inside the graph script. If the string is empty, will try invoking a function called `render_graph_<graphName>()`
            \return A new render-graph object or nullptr if something went horribly wrong
        */
        static RenderGraph::SharedPtr import(std::string graphName, std::filesystem::path path = {}, std::string funcName = {});

        /** Import all the graphs found in the script's global namespace
        */
        static std::vector<RenderGraph::SharedPtr> importAllGraphs(const std::filesystem::path& path);
    };

    class FALCOR_API RenderGraphExporter
    {
    public:
        static std::string getIR(const RenderGraph::SharedPtr& pGraph);
        static std::string getFuncName(const std::string& graphName);
        static bool save(const RenderGraph::SharedPtr& pGraph, std::filesystem::path path = {});
    };
}
