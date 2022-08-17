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

/** Requirements to use this pass:
    1) Have the OptiX 7.3 SDK installed (directly or via packman)
    2) Have NVIDIA driver 465.84 or later.

    When porting this pass, especially to older Falcor forks, it sometimes becomes
    dependent on the DLL cudart64_101.dll, which is generally not copied into the binary
    directory.  Depending on the version of Falcor, not finding all DLL dependencies
    causes Mogwai to crash mysteriously in loadLibrary() when loading a render pass DLL.
*/

/** Usage:

    A simple encapsulation of the OptiX Denoiser. It is not guaranteed optimal.
    In fact, it is definitely suboptimal, as it targets flexibility to use in *any*
    Falcor render graph without awareness of any DX <-> OptiX interop requirements.
    The pass uses resource copies that could be optimized away, adding some overhead,
    though on a RTX 3090, this pass takes about 3ms at 1080p, which seems quite reasonable.

    Using this pass:
     * Connect noisy color image to the "color" pass texture
          - Can be LDR or HDR.  My testing shows the HDR model works just fine
            on LDR inputs... so this pass defaults to using HDR.
     * (Optionally) connect non-noisy albedo and normals to the "albedo" and
       "normal" pass inputs.  Think:  These come directly from your G-buffer.
     * (Optionally) connect non-noisy motion vectors to the "mvec" pass input.
       Use image-space motion vectors, as output by the Falcor G-Buffer pass.
     * Denoised results get output to the "output" pass texture
     * Basic UI controls many OptiX settings, though a few are not yet exposed.
     * The following parameters can be used in Python / scripting to control
       startup / initial default settings:
          - model [OptixDenoiserModel.LDR/HDR/Temporal/AOV]  Note: AOVs not yet supported
          - denoiseAlpha [True/False]:  Should denoising run on alpha channel of input?
          - blend [0...1]:  Output a blend of denoised and input (0 = denoised, 1 = noisy)
 */

#pragma once

#include "Falcor.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"

#include "CudaUtils.h"

using namespace Falcor;

// Note: The trailing underscore is to avoid clashing with the OptixDenoiser type in optix.h
class OptixDenoiser_ : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<OptixDenoiser_>;

    static const Info kInfo;

    static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict);

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene) override;

    // Scripting functions
    bool getEnabled() const { return mEnabled; }
    void setEnabled(bool enabled) { mEnabled = enabled; }

private:
    OptixDenoiser_(const Dictionary& dict);

    Scene::SharedPtr mpScene;

    /** Initializes OptiX & CUDA contexts.  Returns true on success (if false, everything else will fail).
    */
    bool initializeOptix();

    /** Call when we need to (re-)create an OptiX denoiser, on initialization or when settings change.
    */
    void setupDenoiser();

    /** The OptiX denoiser expects inputs and outputs as flat arrays (i.e., not CUDA arrays / textures
        with z-order internal memory layout).  We can either bang on CUDA/OptiX to support that *or* we can
        convert layout on the DX size with a pre-/post-pass to convert to a flat array, then share flat
        arrays with OptiX.  While conversion is non-optimal, we'd need to do internal blit()s anyways (to
        avoid exposing OptiX interop outside this render pass) so this isn't much slower than a better-designed
        sharing of GPU memory between DX and OptiX.
    */
    void convertTexToBuf(RenderContext* pRenderContext, const Texture::SharedPtr& tex, const Buffer::SharedPtr& buf, const uint2& size);
    void convertNormalsToBuf(RenderContext* pRenderContext, const Texture::SharedPtr& tex, const Buffer::SharedPtr& buf, const uint2& size, rmcv::mat4 viewIT);
    void convertBufToTex(RenderContext* pRenderContext, const Buffer::SharedPtr& buf, const Texture::SharedPtr& tex, const uint2& size);
    void convertMotionVectors(RenderContext* pRenderContext, const Texture::SharedPtr& tex, const Buffer::SharedPtr& buf, const uint2& size);

    // Options and parameters for the Falcor render pass
    bool                        mEnabled = true;            ///< True = using OptiX denoiser, False = pass is a no-op
    bool                        mSelectBestMode = true;     ///< Will select best mode automatically (changed to false if the mode is set by Python)
    bool                        mIsFirstFrame = true;       ///< True on the first frame after (re-)creating a denoiser
    bool                        mHasColorInput = true;      ///< Do we have a color input?
    bool                        mHasAlbedoInput = false;    ///< Do we have an albedo guide image for denoising?
    bool                        mHasNormalInput = false;    ///< Do we have a normal guide image for denoising?
    bool                        mHasMotionInput = false;    ///< Do we have input motion vectors for temporal denoising?
    uint2                       mBufferSize = uint2(0, 0);  ///< Current window / render size
    bool                        mRecreateDenoiser = true;   ///< Do we need to (re-)initialize the denoiser before invoking it?

    // GUI helpers for choosing between different OptiX AI denoiser modes
    Gui::DropdownList           mModelChoices = {};
    uint32_t                    mSelectedModel = OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR;

    // Optix context
    bool                        mOptixInitialized = false;
    OptixDeviceContext          mOptixContext = nullptr;

    // Structure to encapsulate DX <-> CUDA interop data for a buffer
    struct Interop
    {
        Buffer::SharedPtr       buffer;                       // Falcor buffer
        CUdeviceptr             devicePtr = (CUdeviceptr)0;   // CUDA pointer to buffer
    };

    // Encapsulte our denoiser parameters, settings, and state.
    struct
    {
        // Various OptiX denoiser parameters and handles.  Explicitly initialize everything, just to be sure.
        OptixDenoiserOptions    options = { 0u, 0u };
        OptixDenoiserModelKind  modelKind = OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR;
        OptixDenoiser           denoiser = nullptr;
        OptixDenoiserParams     params = { 0u, static_cast<CUdeviceptr>(0), 0.0f, static_cast<CUdeviceptr>(0) };
        OptixDenoiserSizes      sizes = {};

        // TODO: Parameters currently set to false and not exposed to the user.  These parameters are here to
        // lay the groundwork for more advanced options in the OptiX denoiser, *however* there has not been
        // testing or even validation that all parameters are set correctly to enable these settings.
        bool                    kernelPredictionMode = false;
        bool                    useAOVs = false;
        uint32_t                tileOverlap = 0u;

        // If using tiled denoising (not tested), set appropriately, otherwise set these to the input / output image size.
        uint32_t                tileWidth = 0u;
        uint32_t                tileHeight = 0u;

        // A wrapper around denoiser inputs for guide normals, albedo, and motion vectors
        OptixDenoiserGuideLayer guideLayer = {};

        // A wrapper around denoiser input color, output color, and prior frame's output (for temporal reuse)
        OptixDenoiserLayer      layer = {};

        // A wrapper around our guide layer interop with DirectX
        struct Intermediates
        {
            Interop             normal;
            Interop             albedo;
            Interop             motionVec;
            Interop             denoiserInput;
            Interop             denoiserOutput;
        } interop;

        // GPU memory we need to allocate for the Optix denoiser to play in & store temporaries
        CudaBuffer  scratchBuffer, stateBuffer, intensityBuffer, hdrAverageBuffer;

    } mDenoiser;

    // Our shaders for converting buffers on input and output from OptiX
    ComputePass::SharedPtr      mpConvertTexToBuf;
    ComputePass::SharedPtr      mpConvertNormalsToBuf;
    ComputePass::SharedPtr      mpConvertMotionVectors;
    FullScreenPass::SharedPtr   mpConvertBufToTex;
    Fbo::SharedPtr              mpFbo;

    /** Allocate a DX <-> CUDA staging buffer
    */
    void allocateStagingBuffer(RenderContext* pRenderContext, Interop& interop, OptixImage2D& image, OptixPixelFormat format = OPTIX_PIXEL_FORMAT_FLOAT4);

    /** Not strictly required, but can be used to deallocate a staging buffer if a user toggles its use off
    */
    void freeStagingBuffer(Interop& interop, OptixImage2D& image);

    /** Reallocate all our staging buffers for DX <-> CUDA/Optix interop
    */
    void reallocateStagingBuffers(RenderContext* pRenderContext);

    /** Get a device pointer from a buffer.  This wrapper gracefully handles nullptrs (i.e., if buf == nullptr)
    */
    void* exportBufferToCudaDevice(Buffer::SharedPtr& buf);
};
