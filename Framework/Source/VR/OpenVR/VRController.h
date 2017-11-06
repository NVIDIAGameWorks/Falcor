/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

// This is wrapper class around OpenVR controllers.  It allows you to determine
//     the state of the contoller (button presses), the current position of the 
//     controler, information about its orientation, and whether it is currently
//     tracked correctly.
//
//  Chris Wyman (12/15/2015)

#pragma once
#include "Framework.h"
#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "API/FBO.h"
#include "API/RenderContext.h"
#include "Graphics/Program/Program.h"
#include "API/Texture.h"

// Forward declare OpenVR system class types to remove "openvr.h" dependencies from Falcor headers
namespace vr { 
    class IVRSystem;              // A generic system with basic API to talk with a VR system
    class IVRCompositor;          // A class that composite rendered results with appropriate distortion into the HMD
    class IVRRenderModels;        // A class giving access to renderable geometry for things like controllers
    class IVRChaperone;
    class IVRRenderModels;
    struct TrackedDevicePose_t;   
}

namespace Falcor
{
    // Forward declare the VRSystem class to avoid header cycles.
    class VRSystem;

    /** High-level OpenVR controller abstraction
    */
    class VRController
    {
    public:
        // Declare standard Falcor shared pointer types.
		using SharedPtr = std::shared_ptr<VRController>;
		using SharedConstPtr = std::shared_ptr<const VRController>;

        // Enums for all button bits exposed by OpenVR.  Not all are accessible with Vive Controllers. 
        //     Non-accessible bits are not currently exposed in the enum.
        enum Button : uint64_t
		{
			None       = 0,
			RedButton  = 1ull << 1,
			Grip       = 1ull << 2,
			Touchpad   = 1ull << 32,
			Trigger    = 1ull << 33,
		};

        // Enums for controller button events
        enum class ButtonEvent 
        { 
            None, Press, Unpress, Touch, Untouch 
        };

        // Controller constructor.  Automatically called by other wrapper classes.
        static SharedPtr create( vr::IVRSystem *vrSys, vr::IVRRenderModels *modelClass = 0 );

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Simple accessor methods
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Check if the controller is active an/or tracking.
        bool isControllerActive( void ) const               { return mIsActive; }
        bool isControllerTracking( void ) const             { return mIsTracking; }

        // True if specified button is currently down or touched.
        bool isButtonDown( Button queryButton ) const       { return ( mPressedButtons & uint64_t(queryButton) ) != 0ull; }
        bool isButtonTouched( Button queryButton ) const    { return ( mTouchedButtons & uint64_t(queryButton) ) != 0ull; }

        // True if button state has changed just recently (since last call to isButtonJust*() or hasButtonStateChanged())
        bool isButtonJustDown( Button queryButton );
        bool isButtonJustTouched( Button queryButton );

        // True if queried button state has changed since last calling hasButtonStateChanged() for specified button
        bool hasButtonStateChanged( Button queryButton );

        // Get a matrix transforming from model/object to world coordinates based on tracking data
        glm::mat4 getToWorldMatrix( void ) const            { return mWorldMatrix; }

        // Returns a [0..1] value representing how far the trigger is depressed.  Values < 0.05 are unreliable.  Values > 0.66
        //   only occur with fairly hard grip.  isButtonDown( Trigger ) essentially uses a threshold on this inside OpenVR.
        float getTriggerDepression( void ) const            { return mTriggerSqueeze; }

        // If trackpad is pressed (or touched) returns the position of this press in [-1..1] x [-1..1]
        glm::vec2 getTrackpadPosition( void ) const         { return mTrackpadPosition; }


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Controller positional accessors
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Get the center of the controller in world space
        glm::vec3 getControllerCenter( void ) const         { return mControllerCenter; }

        // Gets a world-space unit vector pointing out of the top of the Vive controller (think: like a sword blade)
        glm::vec3 getControllerVector( void ) const         { return mControllerVector; }

        // Get the world-space location of a user-specified "aim point" in the controllers' coordinate space
        glm::vec3 getControllerAimPoint( void ) const;

        // Set the model-space "aim point."  setControllerAimPoint( vec3(0,0,-1) ) is one unit "above" the 
        //    controller (i.e., in the direction getControllerVector())
        void      setControllerAimPoint( glm::vec3 &atPoint );


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Controller geometric accessors
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Get a Falcor model representing the controller.  All controllers will return the same geometry,
        //    though you *can* pass your own texture in if you'd like to override OpenVR's baked AO texture.
        Model::SharedPtr   getRenderableModel( Texture::SharedPtr overrideTexture = nullptr );

        // When rendering, if you directly need the texture for the model, you can grab it here
        Texture::SharedPtr getRenderableModelTexture( void ) { return mpModelTexture; }

        // Get a wiredframe set of axes to render instead of / under the controllers.
        //    -> TODO:  Make more flexible instead of using Chris' preferred axis style
        Model::SharedPtr   getRenderableAxes( void );


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Wrapper-specific methods;  Shouldn't need to use outside of OpenVR wrapper code, generally.
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // When a button even occurs, call this to update state viewable by above accessors
        void updateOnButtonEvent( uint64_t button, ButtonEvent eventType );

        // When a new controller pose is provided by OpenVR, call this to update state viewable by above
        //     accessors.  deviceID is OpenVR's ID for the objects it tracks.
        void updateOnNewPose( vr::TrackedDevicePose_t *newPose, int32_t deviceID );

        // If you want to manually toggle the mIsActive state, call this.  The state is handled automatically 
        //     by OpenVR pose information, but specific event updates can get passed here.
        void updateActivatedState( bool currentlyActive )   { mIsActive = currentlyActive; }

        // When the controller first activates, there'a a bunch of state info about the controller
        //     OpenVR has that we don't -- the name of the model inside the DLL, and other controller
        //     properties.  When the controller is first activated, calling this method allows us
        //     to know what we're dealing with.  Without calling this, some more advanced properties
        //     of this class (i.e., getting the model geometry) won't work correctly.
        void updateOnActivate( int32_t deviceID );

        // If we need to query OpenVR about this controller, you need the internal OpenVR device ID
        int32_t getDeviceID( void ) const             { return mOpenVRDeviceID; }

    private:
        uint64_t   mChangedButtons;      // Which buttons have recently changed state?
        uint64_t   mPressedButtons;      // Which buttons are currently depressed?
        uint64_t   mTouchedButtons;      // Which buttons are currently touched?

        float      mTriggerSqueeze;      // How much is the trigger squeezed
        glm::vec2  mTrackpadPosition;    // Where is the trackpad touched/pressed?

        bool       mIsActive;            // Is the controller active?
        bool       mIsTracking;          // Is the controller correctly tracking?

        glm::vec3  mControllerCenter;    // The center of the controller
        glm::vec3  mControllerVector;    // The pointing direction of the controller

        glm::mat4  mWorldMatrix;         // The model-to-world matrix for the controller

        glm::vec3  mModelSpaceAimPoint;  // The user-specified aim point (input via setControllerAimPoint()
        glm::vec3  mCurrentAimPoint;     // The aim point in the current controller reference frame

        Model::SharedPtr mpRenderableModel;  // A Falcor renderable model.  Can be NULL if OpenVR has no model for controller!
        Model::SharedPtr mpRenderableAxes;
        Texture::SharedPtr mpModelTexture;

        // Internals for accessing OpenVR state.  Not user accessible
        vr::IVRRenderModels *mpRenderModels;  
        vr::IVRSystem       *mpVrSys;
        int32_t              mOpenVRDeviceID;
        std::string          mModelName;

    };
}
