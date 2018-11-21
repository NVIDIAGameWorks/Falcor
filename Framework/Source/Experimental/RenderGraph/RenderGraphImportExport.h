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
#pragma once

namespace Falcor
{
    class RenderGraph;
    class Fbo;

    class RenderGraphImporter
    {
    public:
        /** Import a graph from a file.
            \param[in] graphName The name of the graph to import
            \param[in] filename  The graphs filename. If the string is empty, the function will search for a file called `<graphName>.py`
            \param[in] funcName  The function name inside the graph script. If the string is empty, will try invoking a function called `render_graph_<graphName>()`
            \return A new render-graph object or nullptr if something went horribly wrong
        */
        static std::shared_ptr<RenderGraph> import(std::string graphName, std::string filename = {}, std::string funcName = {}, const Fbo* pDstFbo = nullptr);

        struct GraphData
        {
            std::string name;
            std::shared_ptr<RenderGraph> pGraph;
        };

        /** Import all the graphs found in the script's global namespace
        */
        static std::vector <GraphData> importAllGraphs(const std::string& filename, const Fbo* pDstFbo = nullptr);
    };

    class RenderGraphExporter
    {
    public:
        enum class ExportFlags
        {
            None,
            SetScene
        };

        static bool save(const std::shared_ptr<RenderGraph>& pGraph, std::string graphName, std::string filename = {}, std::string funcName = {}, ExportFlags exportFlags = ExportFlags::None);
    };
}