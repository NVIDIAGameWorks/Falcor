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
#include "Graphics/RenderGraph/RenderGraph.h"

namespace pybind11
{
    class module;
};

namespace Falcor
{
    class RenderGraphScripting
    {
    public:
        using SharedPtr = std::shared_ptr<RenderGraphScripting>;
        using GraphDesc = Scripting::Context::ObjectDesc<RenderGraph::SharedPtr>;

        using GraphVec = std::vector<GraphDesc>;

        static SharedPtr create();
        static SharedPtr create(const std::string& filename);

        // Python to C++
        static void registerScriptingObjects(pybind11::module& m);
        bool runScript(const std::string& script);
        const GraphVec& getGraphs() const { return mGraphVec; }
        RenderGraph::SharedPtr getGraph(const std::string& name) const;
        void addGraph(const std::string& name, const RenderGraph::SharedPtr& pGraph);

        static const char* kAddPass;
        static const char* kRemovePass;
        static const char* kAddEdge;
        static const char* kRemoveEdge;
        static const char* kMarkOutput;
        static const char* kUnmarkOutput;
        static const char* kAutoGenEdges;
        static const char* kCreatePass;
        static const char* kCreateGraph;
        static const char* kUpdatePass;
        static const char* kSetName;
        static const char* kSetScene;
        static const char* kLoadPassLibrary;
    private:
        RenderGraphScripting() = default;
        Scripting::Context mContext;
        GraphVec mGraphVec;
        std::string mFilename;
    };
}
