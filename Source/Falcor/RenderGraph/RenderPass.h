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
#include "RenderPassReflection.h"
#include "Utils/Scripting/Dictionary.h"
#include "Utils/InternalDictionary.h"
#include "ResourceCache.h"
#include "Core/API/Texture.h"
#include "Scene/Scene.h"
#include "Utils/UI/Gui.h"
#include "Utils/UI/UserInput.h"
#include "Core/API/RenderContext.h"

namespace Falcor
{
    /** Helper class that's passed to the user during `RenderPass::execute()`
    */
    class dlldecl RenderData
    {
    public:
        /** Get a resource
            \param[in] name The name of the pass' resource (i.e. "outputColor"). No need to specify the pass' name
            \return If the name exists, a pointer to the resource. Otherwise, nullptr
        */
        const Resource::SharedPtr& operator[](const std::string& name) const { return getResource(name); }

        /** Get a resource
            \param[in] name The name of the pass' resource (i.e. "outputColor"). No need to specify the pass' name
            \return If the name exists, a pointer to the resource. Otherwise, nullptr
        */
        const Resource::SharedPtr& getResource(const std::string& name) const;

        /** Get the global dictionary. You can use it to pass data between different passes
        */
        InternalDictionary& getDictionary() const { return (*mpDictionary); }

        /** Get the global dictionary. You can use it to pass data between different passes
        */
        InternalDictionary::SharedPtr getDictionaryPtr() const { return mpDictionary; }

        /** Get the default dimensions used for Texture2Ds (when `0` is specified as the dimensions in `RenderPassReflection`)
        */
        const uint2& getDefaultTextureDims() const { return mDefaultTexDims; }

        /** Get the default format used for Texture2Ds (when `Unknown` is specified as the format in `RenderPassReflection`)
        */
        ResourceFormat getDefaultTextureFormat() const { return mDefaultTexFormat; }
    protected:
        friend class RenderGraphExe;
        RenderData(const std::string& passName, const ResourceCache::SharedPtr& pResourceCache, const InternalDictionary::SharedPtr& pDict, const uint2& defaultTexDims, ResourceFormat defaultTexFormat);
        const std::string& mName;
        ResourceCache::SharedPtr mpResources;
        InternalDictionary::SharedPtr mpDictionary;
        uint2 mDefaultTexDims;
        ResourceFormat mDefaultTexFormat;
    };

    /** Base class for render passes.

        Render passes are expected to implement a static create() function that returns
        a shared pointer to a new object, or throws an exception if creation failed.
        The constructor should be private to force creation of shared pointers.

        Render passes are inserted in a render graph, which is executed at runtime.
        Each render pass declares its I/O requirements in the reflect() function,
        and as part of the render graph compilation their compile() function is called.
        At runtime, execute() is called each frame to generate the pass outputs.
    */
    class dlldecl RenderPass : public std::enable_shared_from_this<RenderPass>
    {
    public:
        using SharedPtr = std::shared_ptr<RenderPass>;
        virtual ~RenderPass() = default;

        struct CompileData
        {
            RenderPassReflection connectedResources;
            uint2 defaultTexDims;
            ResourceFormat defaultTexFormat;
        };

        /** Called once before compilation. Describes I/O requirements of the pass.
            The requirements can't change after the graph is compiled. If the IO requests are dynamic, you'll need to trigger compilation of the render-graph yourself.
        */
        virtual RenderPassReflection reflect(const CompileData& compileData) = 0;

        /** Will be called during graph compilation. You should throw an exception in case the compilation failed
        */
        virtual void compile(RenderContext* pContext, const CompileData& compileData) {}

        /** Executes the pass.
        */
        virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) = 0;

        /** Get a dictionary that can be used to reconstruct the object
        */
        virtual Dictionary getScriptingDictionary() { return {}; }

        /** Get a string describing what the pass is doing
        */
        virtual std::string getDesc() = 0;

        /** Render the pass's UI
        */
        virtual void renderUI(Gui::Widgets& widget) {}

        /** Set a scene into the render-pass
        */
        virtual void setScene(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene) {}

        /** Mouse event handler.
            Returns true if the event was handled by the object, false otherwise
        */
        virtual bool onMouseEvent(const MouseEvent& mouseEvent) { return false; }

        /** Keyboard event handler
            Returns true if the event was handled by the object, false otherwise
        */
        virtual bool onKeyEvent(const KeyboardEvent& keyEvent) { return false; }

        /** Called upon hot reload (by pressing F5).
            \param[in] reloaded Resources that have been reloaded.
        */
        virtual void onHotReload(HotReloadFlags reloaded) {}

        /** Get the current pass' name as defined in the graph
        */
        const std::string& getName() const { return mName; }

    protected:
        friend class RenderGraph;
        RenderPass() = default;
        std::string mName;
        std::function<void(void)> mPassChangedCB = [] {};
    };
}
