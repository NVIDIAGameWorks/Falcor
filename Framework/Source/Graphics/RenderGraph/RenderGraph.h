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
#include "RenderGraph.h"

namespace Falcor
{
    class RenderGraph
    {
    public:
        using SharedPtr = std::shared_ptr<RenderGraph>;

        /** Create a new object
        */
        static SharedPtr create();

        /** Add a render-pass. The name has to be unique, otherwise the call will be ignored
        */
        void setRenderPass(const RenderPass::SharedPtr& pPass, const std::string& passName);

        /** Get a render-pass
        */
        const RenderPass::SharedPtr& getRenderPass(const std::string& name) const;

        /** Insert an edge from a render-pass' output into a different render-pass input.
            The render passes must be different, the graph must be a DAG.
            The input/output strings have the format `renderPassName.resourceName`, where the `renderPassName` is the name used in `setRenderPass()` and the `resourceName` is the resource-name as described by the render-pass object
        */
        void addEdge(const std::string& output, const std::string& input);

        /** Check if the graph is ready for execution (all passes inputs/outputs have been initialized correctly, no loops in the graph)
        */
        bool isValid() const;

        /** Execute the graph
        */
        void execute() const;

        /** Set an input resource. The name has the format `renderPassName.resourceName`.
            This is an alias for `getRenderPass(renderPassName)->setInput(resourceName, pResource)`
        */
        void setInput(const std::string& name, const std::shared_ptr<Resource>& pResource);

        /** Set an output resource. The name has the format `renderPassName.resourceName`.
            This is an alias for `getRenderPass(renderPassName)->setOutput(resourceName, pResource)`
        */
        void setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource);

        /** Tells the graph to automatically allocate a render-pass output. Use that to tell the graph which outputs you expect to use after rendering
            The name has the format `renderPassName.resourceName`
            Note that calling `setOutput` with the same name will disable the automatic allocation
        */
        void autoAllocateOutput(const std::string& name);
    private:
        RenderGraph();
        std::unordered_map<std::string, RenderPass::SharedPtr> mpPasses;

    };
}