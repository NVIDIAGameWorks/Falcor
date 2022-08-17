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
#include "ResourceCache.h"
#include "Core/Macros.h"
#include "Core/HotReloadFlags.h"
#include "Core/API/Resource.h"
#include "Core/API/Texture.h"
#include "Utils/InternalDictionary.h"
#include "Utils/Scripting/Dictionary.h"
#include "Utils/UI/Gui.h"
#include <functional>
#include <memory>
#include <string_view>
#include <string>

namespace Falcor
{
    class Scene;

    /** Helper class that's passed to the user during `RenderPass::execute()`
    */
    class FALCOR_API RenderData
    {
    public:
        /** Get a resource
            \param[in] name The name of the pass' resource (i.e. "outputColor"). No need to specify the pass' name
            \return If the name exists, a pointer to the resource. Otherwise, nullptr
        */
        const Resource::SharedPtr& operator[](const std::string_view name) const { return getResource(name); }

        /** Get a resource
            \param[in] name The name of the pass' resource (i.e. "outputColor"). No need to specify the pass' name
            \return If the name exists, a pointer to the resource. Otherwise, nullptr
        */
        const Resource::SharedPtr& getResource(const std::string_view name) const;

        /** Get a texture
            \param[in] name The name of the pass' texture (i.e. "outputColor"). No need to specify the pass' name
            \return If the texture exists, a pointer to the texture. Otherwise, nullptr
        */
        Texture::SharedPtr getTexture(const std::string_view name) const;

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
        RenderData(const std::string& passName, const ResourceCache::SharedPtr& pResourceCache, const InternalDictionary::SharedPtr& pDict, const uint2& defaultTexDims, ResourceFormat defaultTexFormat);

        const std::string& mName;
        ResourceCache::SharedPtr mpResources;
        InternalDictionary::SharedPtr mpDictionary;
        uint2 mDefaultTexDims;
        ResourceFormat mDefaultTexFormat;

        friend class RenderGraphExe;
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
    class FALCOR_API RenderPass
    {
    public:
        using SharedPtr = std::shared_ptr<RenderPass>;
        virtual ~RenderPass() = default;

        // Render pass info.
        struct Info
        {
            std::string type;   ///< Type name of the render pass. In general this should match the name of the class implementing the render pass.
            std::string desc;   ///< Brief textural description of what the render pass does.
        };

        struct CompileData
        {
            uint2 defaultTexDims;                       ///< Default texture dimension (same as the swap chain size).
            ResourceFormat defaultTexFormat;            ///< Default texture format (same as the swap chain format).
            RenderPassReflection connectedResources;    ///< Reflection data for connected resources, if available. This field may be empty when reflect() is called.
        };

        /** Get the render pass info data.
        */
        const Info& getInfo() const { return mInfo; }

        /** Get the render pass type.
        */
        const std::string& getType() const { return mInfo.type; }

        /** Get the render pass description.
        */
        const std::string& getDesc() const { return mInfo.desc; }

        /** Called before render graph compilation. Describes I/O requirements of the pass.
            The function may be called repeatedly and should not perform any expensive operations.
            The requirements can't change after the graph is compiled. If the I/O are dynamic, you'll need to
            trigger re-compilation of the render graph yourself by calling 'requestRecompile()'.
        */
        virtual RenderPassReflection reflect(const CompileData& compileData) = 0;

        /** Will be called during graph compilation. You should throw an exception in case the compilation failed
        */
        virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) {}

        /** Executes the pass.
        */
        virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) = 0;

        /** Get a dictionary that can be used to reconstruct the object
        */
        virtual Dictionary getScriptingDictionary() { return {}; }

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

        /** Applies graph presets to this render pass.
         */
        virtual void applySettings(const Dictionary& dict) {}

        /** Get the current pass' name as defined in the graph
        */
        const std::string& getName() const { return mName; }

    protected:
        RenderPass(const Info& info) : mInfo(info) {}

        /** Request a recompilation of the render graph.
            Call this function if the I/O requirements of the pass have changed.
            During the recompile, reflect() will be called for the pass to report the new requirements.
        */
        void requestRecompile() { mPassChangedCB(); }

        const Info mInfo;
        std::string mName;

        std::function<void(void)> mPassChangedCB = [] {};

        friend class RenderGraph;
    };
}
