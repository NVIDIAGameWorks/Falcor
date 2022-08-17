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
#include "RenderPass.h"
#include "RenderPassReflection.h"
#include "ResourceCache.h"
#include "RenderGraphExe.h"
#include "Core/Macros.h"
#include <string>
#include <utility>
#include <vector>

namespace Falcor
{
    class RenderGraph;

    class FALCOR_API RenderGraphCompiler
    {
    public:
        struct Dependencies
        {
            ResourceCache::DefaultProperties defaultResourceProps;
            ResourceCache::ResourcesMap externalResources;
        };
        static RenderGraphExe::SharedPtr compile(RenderGraph& graph, RenderContext* pRenderContext, const Dependencies& dependencies);

    private:
        RenderGraphCompiler(RenderGraph& graph, const Dependencies& dependencies);
        RenderGraph& mGraph;
        const Dependencies& mDependencies;

        struct PassData
        {
            uint32_t index;
            RenderPass::SharedPtr pPass;
            std::string name;
            RenderPassReflection reflector;
        };
        std::vector<PassData> mExecutionList;

        // TODO Better way to track history, or avoid changing the original graph altogether?
        struct
        {
            std::vector<std::string> generatedPasses;
            std::vector<std::pair<std::string, std::string>> removedEdges;
        } mCompilationChanges;

        void resolveExecutionOrder();
        void compilePasses(RenderContext* pRenderContext);
        bool insertAutoPasses();
        void allocateResources(ResourceCache* pResourceCache);
        void validateGraph() const;
        void restoreCompilationChanges();
        RenderPass::CompileData prepPassCompilationData(const PassData& passData);
    };
}
