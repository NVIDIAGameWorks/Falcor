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
#pragma once
#include "ResourceCache.h"
#include "Utils/InternalDictionary.h"
#include "RenderPass.h"

namespace Falcor
{
    class RenderGraphCompiler;

    class dlldecl RenderGraphExe
    {
    public:
        using SharedPtr = std::shared_ptr<RenderGraphExe>;
        struct Context
        {
            RenderContext* pRenderContext;
            InternalDictionary::SharedPtr pGraphDictionary;
            uint2 defaultTexDims;
            ResourceFormat defaultTexFormat;
        };

        /** Execute the graph
        */
        void execute(const Context& ctx);

        /** Render the UI
        */
        void renderUI(Gui::Widgets& widget);

        /** Mouse event handler.
            Returns true if the event was handled by the object, false otherwise
        */
        bool onMouseEvent(const MouseEvent& mouseEvent);

        /** Keyboard event handler
        Returns true if the event was handled by the object, false otherwise
        */
        bool onKeyEvent(const KeyboardEvent& keyEvent);

        /** Called upon hot reload (by pressing F5).
            \param[in] reloaded Resources that have been reloaded.
        */
        void onHotReload(HotReloadFlags reloaded);

        /** Get a resource from the cache
        */
        Resource::SharedPtr getResource(const std::string& name) const;

        /** Set an external input resource
            \param[in] name Input name. Has the format `renderPassName.resourceName`
            \param[in] pResource The resource to bind. If this is nullptr, will unregister the resource
        */
        void setInput(const std::string& name, const Resource::SharedPtr& pResource);

    private:
        friend class RenderGraphCompiler;
        static SharedPtr create() { return SharedPtr(new RenderGraphExe); }
        RenderGraphExe() = default;

        void insertPass(const std::string& name, const RenderPass::SharedPtr& pPass);

        struct Pass
        {
            std::string name;
            RenderPass::SharedPtr pPass;
        private:
            friend class RenderGraphExe; // Force RenderGraphCompiler to use insertPass() by hiding this Ctor from it
            Pass(const std::string& name_, const RenderPass::SharedPtr& pPass_) : name(name_), pPass(pPass_) {}
        };

        std::vector<Pass> mExecutionList;
        ResourceCache::SharedPtr mpResourceCache;
    };
}
