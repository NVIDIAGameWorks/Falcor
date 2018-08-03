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
#include "RenderPassReflection.h"
#include "RenderPassSerializer.h"
#include "ResourceCache.h"

namespace Falcor
{
    class Scene;
    class Texture;
    class Gui;
    class RenderContext;
    
    class ScratchPad : public std::enable_shared_from_this<ScratchPad>
    {
    public:
        using SharedPtr = std::shared_ptr<ScratchPad>;
    };

    class RenderData
    {
    public:
        RenderData(const std::string& passName, const ScratchPad::SharedPtr pScratchPad, const ResourceCache::SharedPtr& pResourceDepositBox) : mName(passName), mpResources(pResourceDepositBox), mpScratchPad(pScratchPad) {}
        std::shared_ptr<Texture> getTexture(const std::string& name) const
        {
            return std::dynamic_pointer_cast<Texture>(mpResources->getResource(mName + '.' + name));
        }

        ScratchPad* getScratchPad() const { return mpScratchPad.get(); }
    protected:
        const std::string& mName;
        ResourceCache::SharedPtr mpResources;
        ScratchPad::SharedPtr mpScratchPad;
    };

    class RenderPass : public std::enable_shared_from_this<RenderPass>
    {
    public:
        using SharedPtr = std::shared_ptr<RenderPass>;

        /** Called once before compilation. Describes I/O requirements of the pass.
            The requirements can't change after the graph is compiled. If the IO requests are dynamic, you'll need to trigger compilation of the render-graph yourself.
        */
        virtual void reflect(RenderPassReflection& reflector) const = 0;

        /** Executes the pass.
        */
        virtual void execute(RenderContext* pRenderContext, const RenderData* pData) = 0;

        /** Serialize pass into serialization object
        */
        virtual RenderPassSerializer serialize() { return {}; }

        /** Render the pass's UI
        */
        virtual void renderUI(Gui* pGui, const char* uiGroup) {}

        /** Will be called whenever the backbuffer size changed
        */
        virtual void onResize(uint32_t width, uint32_t height) {}

        /** Set a scene into the render-pass
        */
        virtual void setScene(const std::shared_ptr<Scene>& pScene) {}

        /** Mouse event handler.
            Returns true if the event was handled by the object, false otherwise
        */
        virtual bool onMouseEvent(const MouseEvent& mouseEvent) { return false; }

        /** Keyboard event handler
        Returns true if the event was handled by the object, false otherwise
        */
        virtual bool onKeyEvent(const KeyboardEvent& keyEvent) { return false; }

        /** Get the pass' name
        */
        const std::string& getName() const { return mName; }

        using PassChangedCallback = std::function<void(void)>;

        /** Set the callback function
        */
        void setPassChangedCB(PassChangedCallback cb) { mPassChangedCB = cb; }
    protected:
        RenderPass(const std::string& name) : mName(name) 
        {
            auto cb = [] {};
            mPassChangedCB = cb;
        }
        std::string mName;
        PassChangedCallback mPassChangedCB;
    };
}