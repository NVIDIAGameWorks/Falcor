/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "ResourceCache.h"
#include "Core/Macros.h"
#include "Core/HotReloadFlags.h"
#include "Core/API/Formats.h"
#include "Utils/Math/Vector.h"
#include "Utils/UI/Gui.h"
#include "Utils/Dictionary.h"
#include <memory>
#include <string>
#include <vector>

namespace Falcor
{
class RenderGraphCompiler;
class RenderContext;

class FALCOR_API RenderGraphExe
{
public:
    struct Context
    {
        RenderContext* pRenderContext;
        Dictionary& passesDictionary;
        uint2 defaultTexDims;
        ResourceFormat defaultTexFormat;
    };

    /**
     * Execute the graph
     */
    void execute(const Context& ctx);

    /**
     * Render the UI
     */
    void renderUI(RenderContext* pRenderContext, Gui::Widgets& widget);

    /**
     * Mouse event handler.
     * Returns true if the event was handled by the object, false otherwise
     */
    bool onMouseEvent(const MouseEvent& mouseEvent);

    /**
     * Keyboard event handler
     * Returns true if the event was handled by the object, false otherwise
     */
    bool onKeyEvent(const KeyboardEvent& keyEvent);

    /**
     * Called upon hot reload (by pressing F5).
     * @param[in] reloaded Resources that have been reloaded.
     */
    void onHotReload(HotReloadFlags reloaded);

    /**
     * Get a resource from the cache
     */
    ref<Resource> getResource(const std::string& name) const;

    /**
     * Set an external input resource
     * @param[in] name Input name. Has the format `renderPassName.resourceName`
     * @param[in] pResource The resource to bind. If this is nullptr, will unregister the resource
     */
    void setInput(const std::string& name, const ref<Resource>& pResource);

private:
    friend class RenderGraphCompiler;

    void insertPass(const std::string& name, const ref<RenderPass>& pPass);

    struct Pass
    {
        std::string name;
        ref<RenderPass> pPass;

    private:
        friend class RenderGraphExe; // Force RenderGraphCompiler to use insertPass() by hiding this Ctor from it
        Pass(const std::string& name_, const ref<RenderPass>& pPass_) : name(name_), pPass(pPass_) {}
    };

    std::vector<Pass> mExecutionList;
    std::unique_ptr<ResourceCache> mpResourceCache;
};
} // namespace Falcor
