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
#include "OptixDenoiser.h"
#include "CudaUtils.h"
#include "RenderGraph/RenderPassLibrary.h"

const RenderPass::Info OptixDenoiser_::kInfo { "OptixDenoiser", "Apply the OptiX AI Denoiser." };

namespace
{
    // Names for pass input and output textures
    const char kColorInput[] = "color";
    const char kAlbedoInput[] = "albedo";
    const char kNormalInput[] = "normal";
    const char kMotionInput[] = "mvec";
    const char kOutput[] = "output";

    // Names for configuration options available in Python
    const char kEnabled[] = "enabled";
    const char kBlend[] = "blend";
    const char kModel[] = "model";
    const char kDenoiseAlpha[] = "denoiseAlpha";

    // Locations of shaders used to (re-)format data as needed by OptiX
    const std::string kConvertTexToBufFile = "RenderPasses/OptixDenoiser/ConvertTexToBuf.cs.slang";
    const std::string kConvertNormalsToBufFile = "RenderPasses/OptixDenoiser/ConvertNormalsToBuf.cs.slang";
    const std::string kConvertMotionVecFile = "RenderPasses/OptixDenoiser/ConvertMotionVectorInputs.cs.slang";
    const std::string kConvertBufToTexFile = "RenderPasses/OptixDenoiser/ConvertBufToTex.ps.slang";
};

static void regOptixDenoiser(pybind11::module& m)
{
    pybind11::class_<OptixDenoiser_, RenderPass, OptixDenoiser_::SharedPtr> pass(m, "OptixDenoiser");
    pass.def_property(kEnabled, &OptixDenoiser_::getEnabled, &OptixDenoiser_::setEnabled);

    pybind11::enum_<OptixDenoiserModelKind> model(m, "OptixDenoiserModel");
    model.value("LDR", OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_LDR);
    model.value("HDR", OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR);
    model.value("AOV", OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_AOV);
    model.value("Temporal", OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_TEMPORAL);
}

// Don't remove this. it's required for hot-reload to function properly
extern "C" FALCOR_API_EXPORT const char* getProjDir()
{
    return PROJECT_DIR;
}

extern "C" FALCOR_API_EXPORT void getPasses(Falcor::RenderPassLibrary& lib)
{
    lib.registerPass(OptixDenoiser_::kInfo, OptixDenoiser_::create);
    ScriptBindings::registerBinding(regOptixDenoiser);
}

OptixDenoiser_::OptixDenoiser_(const Dictionary& dict)
    : RenderPass(kInfo)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kEnabled) mEnabled = value;
        else if (key == kModel)
        {
            mDenoiser.modelKind = value;
            mSelectBestMode = false;
        }
        else if (key == kBlend) mDenoiser.params.blendFactor = value;
        else if (key == kDenoiseAlpha) mDenoiser.params.denoiseAlpha = (value ? 1u : 0u);
        else logWarning("Unknown field '{}' in a OptixDenoiser dictionary.", key);
    }

    mpConvertTexToBuf = ComputePass::create(kConvertTexToBufFile, "main");
    mpConvertNormalsToBuf = ComputePass::create(kConvertNormalsToBufFile, "main");
    mpConvertMotionVectors = ComputePass::create(kConvertMotionVecFile, "main");
    mpConvertBufToTex = FullScreenPass::create(kConvertBufToTexFile);
    mpFbo = Fbo::create();
}

OptixDenoiser_::SharedPtr OptixDenoiser_::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new OptixDenoiser_(dict));
}

Dictionary OptixDenoiser_::getScriptingDictionary()
{
    Dictionary d;

    d[kEnabled] = mEnabled;
    d[kBlend] = mDenoiser.params.blendFactor;
    d[kModel] = mDenoiser.modelKind;
    d[kDenoiseAlpha] = bool(mDenoiser.params.denoiseAlpha > 0);

    return d;
}

RenderPassReflection OptixDenoiser_::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection r;
    r.addInput(kColorInput, "Color input");
    r.addInput(kAlbedoInput, "Albedo input").flags(RenderPassReflection::Field::Flags::Optional);
    r.addInput(kNormalInput, "Normal input").flags(RenderPassReflection::Field::Flags::Optional);
    r.addInput(kMotionInput, "Motion vector input").flags(RenderPassReflection::Field::Flags::Optional);
    r.addOutput(kOutput, "Denoised output").format(ResourceFormat::RGBA32Float);
    return r;
}

void OptixDenoiser_::setScene(RenderContext* pRenderContext, const std::shared_ptr<Scene>& pScene)
{
    mpScene = pScene;
}

void OptixDenoiser_::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    if (!initializeOptix())
    {
        throw RuntimeError("OptixDenoiser failed to initialize CUDA and/or OptiX!");
    }

    // Determine available inputs
    mHasColorInput = (compileData.connectedResources.getField(kColorInput) != nullptr);
    mHasAlbedoInput = (compileData.connectedResources.getField(kAlbedoInput) != nullptr);
    mHasNormalInput = (compileData.connectedResources.getField(kNormalInput) != nullptr);
    mHasMotionInput = (compileData.connectedResources.getField(kMotionInput) != nullptr);

    // Set correct parameters for the provided inputs.
    mDenoiser.options.guideNormal = mHasNormalInput ? 1u : 0u;
    mDenoiser.options.guideAlbedo = mHasAlbedoInput ? 1u : 0u;

    // If the user specified a denoiser on initialization, respect that.  Otherwise, choose the "best"
    if (mSelectBestMode)
    {
        auto best = mHasMotionInput
            ? OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_TEMPORAL
            : OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR;

        mSelectedModel = best;
        mDenoiser.modelKind = best;
    }

    // Create a dropdown menu for selecting the denoising mode
    mModelChoices = {};
    mModelChoices.push_back({ OPTIX_DENOISER_MODEL_KIND_LDR, "LDR denoising" });
    mModelChoices.push_back({ OPTIX_DENOISER_MODEL_KIND_HDR, "HDR denoising" });
    if (mHasMotionInput)
    {
        mModelChoices.push_back({ OPTIX_DENOISER_MODEL_KIND_TEMPORAL, "Temporal denoising" });
    }

    // (Re-)allocate temporary buffers when render resolution changes
    uint2 newSize = compileData.defaultTexDims;

    // If allowing tiled denoising, these may be smaller than the window size (TODO; not currently handled)
    mDenoiser.tileWidth = newSize.x;
    mDenoiser.tileHeight = newSize.y;

    // Reallocate / reszize our staging buffers for transferring data to and from OptiX / CUDA / DXR
    if (newSize != mBufferSize && newSize.x > 0 && newSize.y > 0)
    {
        mBufferSize = newSize;
        reallocateStagingBuffers(pRenderContext);
    }

    // Size intensity and hdrAverage buffers correctly.  Only one at a time is used, but these are small, so create them both
    if (mDenoiser.intensityBuffer.getSize() != (1 * sizeof(float))) mDenoiser.intensityBuffer.resize(1 * sizeof(float));
    if (mDenoiser.hdrAverageBuffer.getSize() != (3 * sizeof(float))) mDenoiser.hdrAverageBuffer.resize(3 * sizeof(float));

    // Create an intensity GPU buffer to pass to OptiX when appropriate
    if (!mDenoiser.kernelPredictionMode || !mDenoiser.useAOVs)
    {
        mDenoiser.params.hdrIntensity = mDenoiser.intensityBuffer.getDevicePtr();
        mDenoiser.params.hdrAverageColor = static_cast<CUdeviceptr>(0);
    }
    else  // Create an HDR average color GPU buffer to pass to OptiX when appropriate
    {
        mDenoiser.params.hdrIntensity = static_cast<CUdeviceptr>(0);
        mDenoiser.params.hdrAverageColor = mDenoiser.hdrAverageBuffer.getDevicePtr();
    }

    mRecreateDenoiser = true;
}

void OptixDenoiser_::reallocateStagingBuffers(RenderContext* pRenderContext)
{
    // Allocate buffer for our noisy inputs to the denoiser
    allocateStagingBuffer(pRenderContext, mDenoiser.interop.denoiserInput, mDenoiser.layer.input);

    // Allocate buffer for our denoised outputs from the denoiser
    allocateStagingBuffer(pRenderContext, mDenoiser.interop.denoiserOutput, mDenoiser.layer.output);

    // Allocate a guide buffer for our normals (if necessary)
    if (mDenoiser.options.guideNormal > 0)
        allocateStagingBuffer(pRenderContext, mDenoiser.interop.normal, mDenoiser.guideLayer.normal, OPTIX_PIXEL_FORMAT_FLOAT3);
    else
        freeStagingBuffer(mDenoiser.interop.normal, mDenoiser.guideLayer.normal);

    // Allocate a guide buffer for our albedo (if necessary)
    if (mDenoiser.options.guideAlbedo > 0)
        allocateStagingBuffer(pRenderContext, mDenoiser.interop.albedo, mDenoiser.guideLayer.albedo);
    else
        freeStagingBuffer(mDenoiser.interop.albedo, mDenoiser.guideLayer.albedo);

    // Allocate a guide buffer for our motion vectors (if necessary)
    if (mHasMotionInput) // i.e., if using temporal denoising
        allocateStagingBuffer(pRenderContext, mDenoiser.interop.motionVec, mDenoiser.guideLayer.flow, OPTIX_PIXEL_FORMAT_FLOAT2);
    else
        freeStagingBuffer(mDenoiser.interop.motionVec, mDenoiser.guideLayer.flow);
}

void OptixDenoiser_::allocateStagingBuffer(RenderContext* pRenderContext, Interop& interop, OptixImage2D& image, OptixPixelFormat format)
{
    // Determine what sort of format this buffer should be
    uint32_t elemSize = 4 * sizeof(float);
    ResourceFormat falcorFormat = ResourceFormat::RGBA32Float;
    switch (format)
    {
    case OPTIX_PIXEL_FORMAT_FLOAT4:
        elemSize = 4 * sizeof(float);
        falcorFormat = ResourceFormat::RGBA32Float;
        break;
    case OPTIX_PIXEL_FORMAT_FLOAT3:
        elemSize = 3 * sizeof(float);
        falcorFormat = ResourceFormat::RGBA32Float;
        break;
    case OPTIX_PIXEL_FORMAT_FLOAT2:
        elemSize = 2 * sizeof(float);
        falcorFormat = ResourceFormat::RG32Float;
        break;
    default:
        throw RuntimeError("OptixDenoiser called allocateStagingBuffer() with unsupported format");
    }

    // If we had an existing buffer in this location, free it.
    if (interop.devicePtr) freeSharedDevicePtr((void*)interop.devicePtr);

    // Create a new DX <-> CUDA shared buffer using the Falcor API to create, then find its CUDA pointer.
    interop.buffer = Buffer::createTyped(falcorFormat,
        mBufferSize.x * mBufferSize.y,
        Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget | Resource::BindFlags::Shared);
    interop.devicePtr = (CUdeviceptr)exportBufferToCudaDevice(interop.buffer);

    // Setup an OptiXImage2D structure so OptiX will used this new buffer for image data
    image.width = mBufferSize.x;
    image.height = mBufferSize.y;
    image.rowStrideInBytes = mBufferSize.x * elemSize;
    image.pixelStrideInBytes = elemSize;
    image.format = format;
    image.data = interop.devicePtr;
}

void OptixDenoiser_::freeStagingBuffer(Interop& interop, OptixImage2D& image)
{
    // Free the CUDA memory for this buffer, then set our other references to it to NULL to avoid
    // accidentally trying to access the freed memory.
    if (interop.devicePtr) freeSharedDevicePtr((void*)interop.devicePtr);
    interop.buffer = nullptr;
    image.data = static_cast<CUdeviceptr>(0);
}

void OptixDenoiser_::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (mEnabled && mpScene)
    {
        if (mRecreateDenoiser)
        {
            // Sanity checking.  Do not attempt to use temporal denoising without appropriate inputs!
            // If trying to do this, reset model to something sensible.
            if (!mHasMotionInput && mDenoiser.modelKind == OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_TEMPORAL)
            {
                mSelectedModel = OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR;
                mDenoiser.modelKind = OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_HDR;
            }

            // Setup or recreate our denoiser
            setupDenoiser();
            mRecreateDenoiser = false;
            mIsFirstFrame = true;
        }

        // Copy input textures to correct format OptiX images / buffers for denoiser inputs
        // Note: if () conditions are somewhat excessive, due to attempts to track down mysterious, hard-to-repo crashes
        convertTexToBuf(pRenderContext, renderData.getTexture(kColorInput), mDenoiser.interop.denoiserInput.buffer, mBufferSize);
        if (mHasAlbedoInput && mDenoiser.options.guideAlbedo)
        {
            convertTexToBuf(pRenderContext, renderData.getTexture(kAlbedoInput), mDenoiser.interop.albedo.buffer, mBufferSize);
        }
        if (mHasNormalInput && mDenoiser.options.guideNormal)
        {
            convertNormalsToBuf(pRenderContext, renderData.getTexture(kNormalInput), mDenoiser.interop.normal.buffer, mBufferSize, rmcv::transpose(rmcv::inverse(mpScene->getCamera()->getViewMatrix())));
        }
        if (mHasMotionInput && mDenoiser.modelKind == OptixDenoiserModelKind::OPTIX_DENOISER_MODEL_KIND_TEMPORAL)
        {
            convertMotionVectors(pRenderContext, renderData.getTexture(kMotionInput), mDenoiser.interop.motionVec.buffer, mBufferSize);
        }

        // TODO: Find a better way to synchronize
        pRenderContext->flush(true);

        // Compute average intensity, if needed
        if (mDenoiser.params.hdrIntensity)
        {
            optixDenoiserComputeIntensity(
                mDenoiser.denoiser,
                nullptr, // CUDA stream
                &mDenoiser.layer.input,
                mDenoiser.params.hdrIntensity,
                mDenoiser.scratchBuffer.getDevicePtr(),
                mDenoiser.scratchBuffer.getSize()
            );
        }

        // Compute average color, if needed
        if (mDenoiser.params.hdrAverageColor)
        {
            optixDenoiserComputeAverageColor(
                mDenoiser.denoiser,
                nullptr, // CUDA stream
                &mDenoiser.layer.input,
                mDenoiser.params.hdrAverageColor,
                mDenoiser.scratchBuffer.getDevicePtr(),
                mDenoiser.scratchBuffer.getSize()
            );
        }

        // On the first frame with a new denoiser, we have no prior input for temporal denoising.
        //    In this case, pass in our current frame as both the current and prior frame.
        if (mIsFirstFrame)
        {
            mDenoiser.layer.previousOutput = mDenoiser.layer.input;
        }

        // Run denoiser
        optixDenoiserInvoke(mDenoiser.denoiser,
            nullptr,                 // CUDA stream
            &mDenoiser.params,
            mDenoiser.stateBuffer.getDevicePtr(), mDenoiser.stateBuffer.getSize(),
            &mDenoiser.guideLayer,   // Our set of normal / albedo / motion vector guides
            &mDenoiser.layer,        // Array of input or AOV layers (also contains denoised per-layer outputs)
            1u,                      // Nuumber of layers in the above array
            0u,                      // (Tile) Input offset X
            0u,                      // (Tile) Input offset Y
            mDenoiser.scratchBuffer.getDevicePtr(), mDenoiser.scratchBuffer.getSize());

        // Copy denoised output buffer to texture for Falcor to consume
        convertBufToTex(pRenderContext, mDenoiser.interop.denoiserOutput.buffer, renderData.getTexture(kOutput), mBufferSize);

        // Make sure we set the previous frame output to the correct location for future frames.
        // Everything in this if() cluase could happen every frame, but is redundant after the first frame.
        if (mIsFirstFrame)
        {
            // Note: This is a deep copy that can dangerously point to deallocated memory when resetting denoiser settings.
            // This is (partly) why in the first frame, the layer.previousOutput is set to layer.input, above.
            mDenoiser.layer.previousOutput = mDenoiser.layer.output;

            // We're no longer in the first frame of denoising; no special processing needed now.
            mIsFirstFrame = false;
        }
    }
    else // Denoiser not enabled; copy the noisy input texture to the output
    {
        pRenderContext->blit(renderData.getTexture(kColorInput)->getSRV(), renderData.getTexture(kOutput)->getRTV());
    }
}

void OptixDenoiser_::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Use OptiX Denoiser?", mEnabled);

    if (mEnabled)
    {
        if (widget.dropdown("Model", mModelChoices, mSelectedModel))
        {
            mDenoiser.modelKind = static_cast<OptixDenoiserModelKind>(mSelectedModel);
            mRecreateDenoiser = true;
        }
        widget.tooltip("Selects the OptiX denosing model.  See OptiX documentation for details.\n\nFor best results:\n   LDR assumes inputs [0..1]\n   HDR assumes inputs [0..10,000]");

        if (mHasAlbedoInput)
        {
            bool useAlbedoGuide = mDenoiser.options.guideAlbedo != 0u;
            if (widget.checkbox("Use albedo guide?", useAlbedoGuide))
            {
                mDenoiser.options.guideAlbedo = useAlbedoGuide ? 1u : 0u;
                mRecreateDenoiser = true;
            }
            widget.tooltip("Use input, noise-free albedo channel to help guide denoising.");
        }

        if (mHasNormalInput)
        {
            bool useNormalGuide = mDenoiser.options.guideNormal != 0u;
            if (widget.checkbox("Use normal guide?", useNormalGuide))
            {
                mDenoiser.options.guideNormal = useNormalGuide ? 1u : 0u;
                mRecreateDenoiser = true;
            }
            widget.tooltip("Use input, noise-free normal buffer to help guide denoising.  (Note: The Optix 7.3 API is unclear on this point, but, correct use of normal guides appears to also require using an albedo guide.)");
        }

        {
            bool denoiseAlpha = mDenoiser.params.denoiseAlpha != 0;
            if (widget.checkbox("Denoise Alpha?", denoiseAlpha))
            {
                mDenoiser.params.denoiseAlpha = denoiseAlpha ? 1u : 0u;
            }
            widget.tooltip("Denoise the alpha channel, not just RGB.");
        }

        widget.slider("Blend", mDenoiser.params.blendFactor, 0.f, 1.f);
        widget.tooltip("Blend denoised and original input. (0 = denoised only, 1 = noisy only)");
    }
}


// Basically a wrapper to handle null Falcor Buffers gracefully, which couldn't
// happen in getShareDevicePtr(), due to the bootstrapping that avoids namespace conflicts
void * OptixDenoiser_::exportBufferToCudaDevice(Buffer::SharedPtr &buf)
{
    if (buf == nullptr) return nullptr;
    return getSharedDevicePtr(buf->getSharedApiHandle(), (uint32_t)buf->getSize());
}

bool OptixDenoiser_::initializeOptix()
{
    if (!mOptixInitialized)
    {
        mOptixInitialized = initOptix(mOptixContext);
    }
    return mOptixInitialized;
}

void OptixDenoiser_::setupDenoiser()
{
    // Destroy the denoiser, if it already exists
    if (mDenoiser.denoiser)
    {
        optixDenoiserDestroy(mDenoiser.denoiser);
    }

    // Create the denoiser
    optixDenoiserCreate(mOptixContext,
        mDenoiser.modelKind,
        &mDenoiser.options,
        &mDenoiser.denoiser);

    // Find out how much memory is needed for the requested denoiser
    optixDenoiserComputeMemoryResources(mDenoiser.denoiser, mDenoiser.tileWidth, mDenoiser.tileHeight, &mDenoiser.sizes);

    // Allocate/resize some temporary CUDA buffers for internal OptiX processing/state
    mDenoiser.scratchBuffer.resize(mDenoiser.sizes.withoutOverlapScratchSizeInBytes);
    mDenoiser.stateBuffer.resize(mDenoiser.sizes.stateSizeInBytes);

    // Finish setup of the denoiser
    optixDenoiserSetup(mDenoiser.denoiser,
        nullptr,
        mDenoiser.tileWidth + 2 * mDenoiser.tileOverlap,   // Should work with tiling if parameters set appropriately
        mDenoiser.tileHeight + 2 * mDenoiser.tileOverlap,  // Should work with tiling if parameters set appropriately
        mDenoiser.stateBuffer.getDevicePtr(), mDenoiser.stateBuffer.getSize(),
        mDenoiser.scratchBuffer.getDevicePtr(), mDenoiser.scratchBuffer.getSize());
}

void OptixDenoiser_::convertMotionVectors(RenderContext* pRenderContext, const Texture::SharedPtr& tex, const Buffer::SharedPtr& buf, const uint2& size)
{
    auto vars = mpConvertMotionVectors->getVars();
    vars["GlobalCB"]["gStride"] = size.x;
    vars["GlobalCB"]["gSize"] = size;
    vars["gInTex"] = tex;
    vars["gOutBuf"] = buf;
    mpConvertMotionVectors->execute(pRenderContext, size.x, size.y);
}

void OptixDenoiser_::convertTexToBuf(RenderContext* pRenderContext, const Texture::SharedPtr& tex, const Buffer::SharedPtr& buf, const uint2& size)
{
    auto vars = mpConvertTexToBuf->getVars();
    vars["GlobalCB"]["gStride"] = size.x;
    vars["gInTex"] = tex;
    vars["gOutBuf"] = buf;
    mpConvertTexToBuf->execute(pRenderContext, size.x, size.y);
}

void OptixDenoiser_::convertNormalsToBuf(RenderContext* pRenderContext, const Texture::SharedPtr& tex, const Buffer::SharedPtr& buf, const uint2& size, rmcv::mat4 viewIT)
{
    auto vars = mpConvertNormalsToBuf->getVars();
    vars["GlobalCB"]["gStride"] = size.x;
    vars["GlobalCB"]["gViewIT"] = viewIT;
    vars["gInTex"] = tex;
    vars["gOutBuf"] = buf;
    mpConvertTexToBuf->execute(pRenderContext, size.x, size.y);
}

void OptixDenoiser_::convertBufToTex(RenderContext* pRenderContext, const Buffer::SharedPtr& buf, const Texture::SharedPtr& tex, const uint2& size)
{
    auto vars = mpConvertBufToTex->getVars();
    vars["GlobalCB"]["gStride"] = size.x;
    vars["gInBuf"] = buf;
    mpFbo->attachColorTarget(tex, 0);
    mpConvertBufToTex->execute(pRenderContext, mpFbo);
}
