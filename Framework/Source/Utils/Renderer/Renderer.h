/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "Falcor.h"

using namespace Falcor;

/** Abstract base class for encapsulated renderers used for for Falcor's MultiRendererSample
*/
class Renderer : public std::enable_shared_from_this<Renderer>
{
public:
	using SharedPtr      = std::shared_ptr<Renderer>;
	using SharedConstPtr = std::shared_ptr<const Renderer>;

	/** Constructor
	    \param[in] rendererName The name used in the MultiRendererSample drop down to select between renderers
	    \param[in] guiName The name of the GUI window
	*/
	Renderer( const std::string rendererName = "<Unknown Renderer>", 
		      const std::string guiName      = "<Unknown Renderer>" ) : mRendererName(rendererName), mGuiGroupName(guiName) {}
	virtual ~Renderer() = default;

	/** Callback on program initialization
		\param[in] pContext Provides the current context to initialize resources for your renderer
	*/
	virtual void onInitialize(RenderContext::SharedPtr pContext) = 0;

	/** Callback on scene load
		\param[in] pSceneRenderer Provides the newly loaded scene (if you need to update resources or stash the scene)
	*/
	virtual void onInitNewScene(SceneRenderer::SharedPtr pSceneRenderer) { }

	/** Callback on GUI render.  Allows addition of renderer-specific UI widgets into the widget window.
	*/
	virtual void onGuiRender() { }

	/** Callback for rendering.  Called whenever a display refresh requested. 
	    \param[in] pContext Provides the current context to initialize resources for your renderer
		\param[in] pTargetFbo The framebuffer where the app expects the final rendering.
	*/
	virtual void onFrameRender(RenderContext::SharedPtr pContext, Fbo::SharedPtr pTargetFbo) = 0;

	/** Callback when the image/resources need to be resized.  (Also called during initialization)
		\param[in] width The new width or your back-buffer
		\param[in] height The new width or your back-buffer
	*/
	virtual void onResizeSwapChain(uint32_t width, uint32_t height) { }

	/** Callback executed when switching to this renderer.  Called when a user switches between renderers
		via the UI (or upon initialization when starting the default renderer)
	*/
	virtual void onSwitchTo() { }

	/** Callback executed when switching from this renderer.  Called when a user switches between renderers
		via the UI (or upon shutdown, before destroying the current renderer)
	*/
	virtual void onSwitchFrom() { }

	/** Callback executed when closing the application.  
	*/
	virtual void onShutdown() { }
	
	/** Callback executed when processing a key event.  Only gets called for the currently selected renderer
		\param[in] keyEvent A Falcor structure containing details about the event
		\return true if we process the key event, false if someone else should
	*/
	virtual bool onKeyEvent(const KeyboardEvent& keyEvent) { return false; }

	/** Callback executed when processing a mouse event.  Only gets called for the currently selected renderer
		\param[in] mouseEvent A Falcor structure containing details about the event
		\return true if we process the mouse event, false if someone else should
	*/
	virtual bool onMouseEvent(const MouseEvent& mouseEvent) { return false; }

	/** Used by the wrapper app to get information about the renderer.  Shouldn't need to overload or touch these.
	*/
	std::string getRendererName() { return mRendererName; }
	std::string getGuiGroupName() { return mGuiGroupName; }
	void setRendererName(const std::string &name) { mRendererName = name; }
	void setGuiGroupName(const std::string &name) { mGuiGroupName = name; }

// Internal state that may or may not prove useful for derived renderers	
protected:
	Gui*						 mpGui = nullptr;                        ///< Falcor GUI
	GraphicsState::SharedPtr     mpState = nullptr;                      ///< Falcor state pointer
	std::string                  mRendererName = "<Unknown Renderer>";   ///< Name used in GUI dropdown to select renderer
	std::string                  mGuiGroupName = "<Unknown Renderer>";   ///< Name used on GUI group for renderer settings

	float                        mCurrentTime   = 0.0f;                  ///< Current time, passed in each frame by the main app
	bool                         mIsInitialized = false;                 ///< Can use this to track if you've already called onInitialize()

	glm::ivec2                   mGuiWinTopLeft = ivec2(20, 300);        ///< Default location for the GUI window
	glm::ivec2                   mGuiSize       = ivec2(250, 300);       ///< Default size of the GUI window




///////////////////////////////////////////////////////////////////////////////////////
// NOTE: 
//    When deriving from the Renderer class, you should largely be able to ignore all
//    methods below this point.  Methods below here are largely used by the main app
//    to cleanly and seamlessly switch between derived renderers.  I should probably
//    encapsulate them better to hide them, but some (like some of the state setters)
//    need to be overridden by more complex Renderers (e.g., those that show two
//    other renderers side-by-side).  So, basically: mess with these at your own risk!
///////////////////////////////////////////////////////////////////////////////////////
public:

	// Deleted copy operators (copy a pointer type!)
	Renderer(const Renderer&) = delete;
	Renderer& operator=(const Renderer &) = delete;

	/** The global app tracks the global time.  This is a setter called by the app to ensure renderers have access to the time
	*/
	void setCurrentTime(float newTime) { mCurrentTime = newTime; }

	/** Get information about the location of where the GUI window should be
	*/
	ivec2 getGuiWindowLocation(void) { return mGuiWinTopLeft; }
	ivec2 getGuiWindowSize(void) { return mGuiSize; }

	/** Should the caller ensure this renderer's UI widgets get their own UI window?  If you do not want a separate
		UI window for widgets created by your renderer, overload this and return false.
	*/
	virtual bool useUIWindow( void ) { return true; }


	/** This is guaranteed to be the first call of any of the virtual callbacks defined above, to set important internal state.
		Note that this order guarantee is only within this renderer (i.e., other virtual callbacks in Renderer B may be called
		before initializizeSharedState() in Renderer A).
	*/
	virtual void initializeSharedState(Gui::UniquePtr& pGui, GraphicsState::SharedPtr pGfxState) { mpGui = pGui.get(); mpState = pGfxState; }

};
