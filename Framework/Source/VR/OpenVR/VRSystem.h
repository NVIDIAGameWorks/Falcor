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

// This is the main class for incorporating basic VR support into your application
//     via OpenVR.  This is currently tested only on the HTC Vive headset.
//
// Basic initialization:
//     VRSystem::UniquePtr vrSys = VRSystem::create( appRenderContext );
//
// Basic render loop:
//     if (vrSys->isReady()) {
//         vrSys->pollEvents();                               // Check for button presses
//         vrSys->refreshTracking();                          // Update HMD/controller positions
//         RenderEyeImages( &myLeftEyeFbo, &myRightEyeFbo );  // Do your rendering
//         vrSys->submit( Eye::Left, myLeftEyeFbo );          // Submit images to HMD
//         vrSys->submit( Eye::Right, myRightEyeFbo );
//     }
//
// Objects for the controller, hmd, tracker, and room are instantiated by this main 
// wrapper and are available via the appropriate get*() accessors.  These objects allow
// access to more specific details needed for rendering. 
//
//
//  Chris Wyman (12/15/2015)

#pragma once

#include "Framework.h"
#include "Graphics/Model/Model.h"
#include "Graphics/Model/Loaders/SimpleModelImporter.h"
#include "glm/mat4x4.hpp"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "API/FBO.h"
#include "API/RenderContext.h"
#include "Graphics/Program/Program.h"
#include "API/Texture.h"
#include "VRController.h"
#include "VRTrackerBox.h"
#include "VRDisplay.h"
#include "VRPlayArea.h"

#pragma comment(lib, "openvr_api.lib")

// Forward declare OpenVR system class types to remove "openvr.h" dependencies from the headers
namespace vr { 
    class IVRSystem;              
    class IVRCompositor;          
    class IVRRenderModels;        
    class IVRChaperone;
    struct TrackedDevicePose_t;   
}

namespace Falcor
{
    /** High-level abstraction for a basic interface to OpenVR
    */
    class VRSystem 
    {
    public:
        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Constructor and destructor methods 
        //////////////////////////////////////////////////////////////////////////////////////////////////

        /** Create an OpenVR system class and initialize the system.
            If using Vulkan, this function needs to be called before creating VkInstance, so that we can retrieve the required extensions.
            For D3D12 this can be called after the device was created
        */
        static VRSystem* start( bool enableVSync = true );
        static VRSystem* instance() { checkInit(); return spVrSystem; }

		// If you want to clean up resources and shut down the VR system, call cleanup(), which acts as
		//     a destructor.  This wrapper does not have an explicit destructor, since Falcor appears
		//     to do something under the hood with such a destructor that either destroys things twice 
		//     or destroys them in an inappropriate order when exiting the program.  (TODO: Debug?)
		static void cleanup(void);

        /** Initialize the display and the controllers
        */
        void initDisplayAndController(RenderContext::SharedPtr pRenderContext);

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Simple accessors
        //////////////////////////////////////////////////////////////////////////////////////////////////
       
        // Is the VR HMD initialized and read to render?  If false, go grab errors via getError()
        bool isReady();

        // Is VSync currently enabled?  (TODO: Add toggle.  But mostly you just want vSync always ON.)
        bool isVSyncEnabled() const { return mVSyncEnabled; }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Basic wrapper usage methods
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Query the HMD for the current position of tracked devices
        void refreshTracking();

        // Poll for events (device [de]activate, button presses, etc).  Returns true if event(s) occurs.
        //     Updates internal state (i.e., in the HMD, controller, tracker classes) to reflect these events
        bool pollEvents();

        // Refresh the HMD. This is just a wrapper which calls refreshTracking() and pollEvents()
        void refresh();

        // Submit rendered frames to the HMD.  Should submit one image to left eye and one to right eye 
        //     each frame.  The submit() routines handle all warping due to lens and color distortions.
        //     Size of each texture should be:  getHMD()->getRecommendedRenderSize() for best results.
        bool submit( VRDisplay::Eye whichEye, const Texture::SharedConstPtr& pDisplayTex, RenderContext* pRenderCtx);


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Access current world state (positions of HMD, controllers, etc) 
        //////////////////////////////////////////////////////////////////////////////////////////////////

        // Get our head mounted display object.  Includes rendering matrices and target size.
        VRDisplay::SharedPtr    getHMD() { return mDisplay; }

        // Get our controller(s).  Includes position, button state, aim, etc.
        VRController::SharedPtr getController( uint32_t idx ) { return (idx < 2) ? mControllers[idx] : nullptr; }

        // Get our tracker(s)/lighthouse(s).  
        VRTrackerBox::SharedPtr getTracker( uint32_t idx ) { return (idx < 2) ? mTracker[idx] : nullptr; }

        // Get information about our room/play area.
        VRPlayArea::SharedPtr   getPlayArea() { return mPlayArea; }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Access renderable geometry.  These are shortcuts (can also be accessed through controller, hmd,
        //     tracker classes).  Can be nullptr if devices haven't been activated or the OpenVR DLL has
        //     not built-in model for them.
        //////////////////////////////////////////////////////////////////////////////////////////////////

        Model::SharedPtr getControllerModel() { return mpControlModel; }
        Model::SharedPtr getTrackerModel() { return mpLighthouseModel; }
        Model::SharedPtr getHMDModel() { return mpHMDModel; }

        // This is an extra, added by Chris, for a simple xyz axis model for controllers.
        Model::SharedPtr getAxesModel() { return mpAxisModel; }

        // Gets a very simple Falcor model representing the HMD's lens distortion.
        //    -> AttirbuteLocation 'Position' is position.  2 components directly representing NDC.
        //    -> AttirbuteLocation 'User0' is the red distortion.  (I.e., offset into original texture for red channel)
        //    -> AttirbuteLocation 'User1' is the green distortion.  (I.e., offset into original texture for green channel)
        //    -> AttirbuteLocation 'User2' is the blue distortion.  (I.e., offset into original texture for blue channel)
        // Note: This model currently cannot be used with Falcor's ModelRenderer.  Why needs to be debugged.
        Model::SharedPtr getDistortionModel( VRDisplay::Eye whichEye ) { return mpLensDistortModel[(whichEye == VRDisplay::Eye::Left) ? 0 : 1]; }


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Error routines
        //////////////////////////////////////////////////////////////////////////////////////////////////
        
        // Pretty rudimentary error messages.  Mostly useful for initialization/constructor failures, currently.
        uint32_t getError( std::string *errMessage = NULL );


        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Enable for low level interactions with OpenVR
        //////////////////////////////////////////////////////////////////////////////////////////////////

		//vr::IVRSystem     *getOpenVRSystemPtr(void) const             { return mpHMD; }
		vr::IVRCompositor *getOpenVRCompositor(void) const            { return mpCompositor; }

        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Display/Timing specific methods
        //////////////////////////////////////////////////////////////////////////////////////////////////
        bool getTimeSinceLastVsync(float *pfSecondsSinceLastVsync, uint64_t *pulFrameCounter);

#ifdef FALCOR_VK
        static std::vector<std::string> getRequiredVkInstanceExtensions();
        static std::vector<std::string> getRequiredVkDeviceExtensions(VkPhysicalDevice device);
#endif
    private:
        //////////////////////////////////////////////////////////////////////////////////////////////////
        // Declare private stuff..
        //////////////////////////////////////////////////////////////////////////////////////////////////

        VRSystem();  
        static VRSystem* spVrSystem;

		// OpenGl/ DirectX context handle from Falcor
		RenderContext::SharedPtr mpContext;

        // Returns a controller/tracker from an OpenVR device ID.  Returns nullptr if unknown deviceID.
        VRController::SharedPtr getControllerFromDeviceID( int32_t devID );
        VRTrackerBox::SharedPtr getTrackerFromDeviceID( int32_t devID );

        // System parameters
        bool       mReadyToRender;
        bool       mVSyncEnabled;
        uint32_t   mRenderAPI;

        // Base OpenVR system objects
        vr::IVRSystem       *mpHMD;
        vr::IVRCompositor   *mpCompositor;
        vr::IVRRenderModels *mpModels;
        vr::IVRChaperone    *mpChaperone;

        // A place to stash OpenVR's most recent error information
        uint32_t           mLastError;  // TODO:  convert to enum
        std::string        mLastErrorMsg;

        // An array of poses we'll pass to OpenVR internally;
        vr::TrackedDevicePose_t *mDevicePoses = nullptr;

        // Some experimental display stuff concerning the HMD's image distortion 
        void createDistortionVBO();

        // Our sub-classes that store state for controllers, hmds, trackers, and the play area
        VRController::SharedPtr mControllers[2];
        VRTrackerBox::SharedPtr mTracker[2];
        VRDisplay::SharedPtr    mDisplay;
        VRPlayArea::SharedPtr   mPlayArea;

        // Renderable models.  We initialize these as soon as we know we have valid instances 
        //     of controllers, trackers, and HMDs.  This makes it a bit easier than querying 
        //     repeatedly or manually processing events.
        Model::SharedPtr  mpControlModel;
        Model::SharedPtr  mpLighthouseModel;
        Model::SharedPtr  mpAxisModel;
        Model::SharedPtr  mpHMDModel;

        // A representation of the lens distortion for the 2 eyes (either to display on screen, or
        //     to render to the HMD if you're not using the IVRSystem's compositor.  However, this 
        //     wrapper only exposes the IVRSystem's compositor right now.
        Model::SharedPtr  mpLensDistortModel[2];

#ifdef _DEBUG
        static void checkInit();
#else
        static void checkInit() {};
#endif
    };
}
