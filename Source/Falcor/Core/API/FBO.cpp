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
#include "FBO.h"
#include "Device.h"
#include "GFXAPI.h"
#include "Core/Error.h"
#include "Core/ObjectPython.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
namespace
{
void releaseFramebuffer(
    Device& device,
    Slang::ComPtr<gfx::IFramebuffer>& framebuffer,
    Fbo::Attachment& depthStencil,
    std::vector<Fbo::Attachment>& colorAttachments
)
{
    if (!framebuffer)
    {
        return;
    }
    if (depthStencil.pTexture)
    {
        return;
    }
    for (auto& attachment : colorAttachments)
    {
        if (attachment.pTexture)
        {
            return;
        }
    }
    device.releaseResource(framebuffer);
    framebuffer.setNull();
}
} // namespace

static Fbo::DescCache& getGlobalDescCache()
{
    static Fbo::DescCache sCache; // TODO: REMOVEGLOBAL
    return sCache;
}

namespace
{
void checkAttachArguments(const Texture* pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize, bool isDepthAttachment)
{
    if (pTexture == nullptr)
        return;

    FALCOR_CHECK(mipLevel < pTexture->getMipCount(), "'mipLevel' ({}) is out of bounds.", mipLevel);

    if (arraySize != Fbo::kAttachEntireMipLevel)
    {
        FALCOR_CHECK(arraySize != 0, "'arraySize' must not be zero.");
        if (pTexture->getType() == Texture::Type::Texture3D)
        {
            FALCOR_CHECK(
                arraySize + firstArraySlice <= pTexture->getDepth(),
                "'firstArraySlice' ({}) and 'arraySize' ({}) request depth index that is out of bounds.",
                firstArraySlice,
                arraySize
            );
        }
        else
        {
            FALCOR_CHECK(
                arraySize + firstArraySlice <= pTexture->getArraySize(),
                "'frstArraySlice' ({}) and 'arraySize' ({}) request array index that is out of bounds.",
                firstArraySlice,
                arraySize
            );
        }
    }

    if (isDepthAttachment)
    {
        FALCOR_CHECK(isDepthStencilFormat(pTexture->getFormat()), "Depth-stencil texture must have a depth-stencil format.");
        FALCOR_CHECK(
            is_set(pTexture->getBindFlags(), ResourceBindFlags::DepthStencil), "Depth-stencil texture must have the DepthStencil bind flag."
        );
    }
    else
    {
        FALCOR_CHECK(!isDepthStencilFormat(pTexture->getFormat()), "Color texture must not have a depth-stencil format.");
        FALCOR_CHECK(
            is_set(pTexture->getBindFlags(), ResourceBindFlags::RenderTarget), "Color texture must have the RenderTarget bind flag."
        );
    }
}

ref<Texture> createTexture2D(
    ref<Device> pDevice,
    uint32_t w,
    uint32_t h,
    ResourceFormat format,
    uint32_t sampleCount,
    uint32_t arraySize,
    uint32_t mipLevels,
    ResourceBindFlags flags
)
{
    if (format == ResourceFormat::Unknown)
    {
        FALCOR_THROW("Can't create Texture2D with an unknown resource format.");
    }

    if (sampleCount > 1)
    {
        return pDevice->createTexture2DMS(w, h, format, sampleCount, arraySize, flags);
    }
    else
    {
        return pDevice->createTexture2D(w, h, format, arraySize, mipLevels, nullptr, flags);
    }
}

ResourceBindFlags getBindFlags(bool isDepth, bool allowUav)
{
    ResourceBindFlags flags = ResourceBindFlags::ShaderResource;
    flags |= isDepth ? ResourceBindFlags::DepthStencil : ResourceBindFlags::RenderTarget;

    if (allowUav)
    {
        flags |= ResourceBindFlags::UnorderedAccess;
    }
    return flags;
}
}; // namespace

size_t Fbo::DescHash::operator()(const Fbo::Desc& d) const
{
    size_t hash = 0;
    std::hash<uint32_t> u32hash;
    std::hash<bool> bhash;
    for (uint32_t i = 0; i < getMaxColorTargetCount(); i++)
    {
        uint32_t format = (uint32_t)d.getColorTargetFormat(i);
        format <<= i;
        hash |= u32hash(format) >> i;
        hash |= bhash(d.isColorTargetUav(i)) << i;
    }

    uint32_t format = (uint32_t)d.getDepthStencilFormat();
    hash |= u32hash(format);
    hash |= bhash(d.isDepthStencilUav());
    hash |= u32hash(d.getSampleCount());

    return hash;
}

bool Fbo::Desc::operator==(const Fbo::Desc& other) const
{
    if (mColorTargets.size() != other.mColorTargets.size())
        return false;

    for (size_t i = 0; i < mColorTargets.size(); i++)
    {
        if (mColorTargets[i] != other.mColorTargets[i])
            return false;
    }
    if (mDepthStencilTarget != other.mDepthStencilTarget)
        return false;
    if (mSampleCount != other.mSampleCount)
        return false;

    return true;
}

Fbo::Desc::Desc()
{
    mColorTargets.resize(Fbo::getMaxColorTargetCount());
}

ref<Fbo> Fbo::create(ref<Device> pDevice)
{
    return ref<Fbo>(new Fbo(pDevice));
}

ref<Fbo> Fbo::create(ref<Device> pDevice, const std::vector<ref<Texture>>& colors, const ref<Texture>& pDepth)
{
    auto pFbo = create(pDevice);
    for (uint32_t i = 0; i < colors.size(); i++)
    {
        pFbo->attachColorTarget(colors[i], i);
    }
    if (pDepth)
    {
        pFbo->attachDepthStencilTarget(pDepth);
    }
    pFbo->finalize();
    return pFbo;
}

Fbo::Fbo(ref<Device> pDevice) : mpDevice(pDevice)
{
    mColorAttachments.resize(getMaxColorTargetCount());
}

Fbo::~Fbo()
{
    mpDevice->releaseResource(mGfxFramebuffer);
}

gfx::IFramebuffer* Fbo::getGfxFramebuffer() const
{
    if (mHandleDirty)
    {
        initFramebuffer();
    }
    return mGfxFramebuffer;
}

uint32_t Fbo::getMaxColorTargetCount()
{
    return 8;
}

void Fbo::applyColorAttachment(uint32_t rtIndex)
{
    mHandleDirty = true;
    releaseFramebuffer(*mpDevice, mGfxFramebuffer, mDepthStencil, mColorAttachments);
}

void Fbo::applyDepthAttachment()
{
    mHandleDirty = true;
    releaseFramebuffer(*mpDevice, mGfxFramebuffer, mDepthStencil, mColorAttachments);
}

void Fbo::initFramebuffer() const
{
    mHandleDirty = false;

    gfx::IFramebufferLayout::Desc layoutDesc = {};
    std::vector<gfx::IFramebufferLayout::TargetLayout> targetLayouts;
    gfx::IFramebufferLayout::TargetLayout depthTargetLayout = {};
    gfx::IFramebuffer::Desc desc = {};
    if (mDepthStencil.pTexture)
    {
        auto gfxTexture = mDepthStencil.pTexture->getGfxTextureResource();
        depthTargetLayout.format = gfxTexture->getDesc()->format;
        depthTargetLayout.sampleCount = gfxTexture->getDesc()->sampleDesc.numSamples;
        layoutDesc.depthStencil = &depthTargetLayout;
    }
    desc.depthStencilView = getDepthStencilView()->getGfxResourceView();
    desc.renderTargetCount = 0;
    std::vector<gfx::IResourceView*> renderTargetViews;
    for (uint32_t i = 0; i < static_cast<uint32_t>(mColorAttachments.size()); i++)
    {
        gfx::IFramebufferLayout::TargetLayout renderTargetLayout = {};

        if (mColorAttachments[i].pTexture)
        {
            auto gfxTexture = mColorAttachments[i].pTexture->getGfxTextureResource();
            renderTargetLayout.format = gfxTexture->getDesc()->format;
            renderTargetLayout.sampleCount = gfxTexture->getDesc()->sampleDesc.numSamples;
            targetLayouts.push_back(renderTargetLayout);
            desc.renderTargetCount = i + 1;
        }
        else
        {
            renderTargetLayout.format = gfx::Format::R8G8B8A8_UNORM;
            renderTargetLayout.sampleCount = 1;
            targetLayouts.push_back(renderTargetLayout);
        }
        renderTargetViews.push_back(getRenderTargetView(i)->getGfxResourceView());
    }
    desc.renderTargetViews = renderTargetViews.data();
    layoutDesc.renderTargetCount = desc.renderTargetCount;
    layoutDesc.renderTargets = targetLayouts.data();

    // Push FBO handle to deferred release queue so it remains valid
    // for pending GPU commands.
    mpDevice->releaseResource(mGfxFramebuffer);

    Slang::ComPtr<gfx::IFramebufferLayout> fboLayout;
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createFramebufferLayout(layoutDesc, fboLayout.writeRef()));

    desc.layout = fboLayout.get();
    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createFramebuffer(desc, mGfxFramebuffer.writeRef()));
}

ref<RenderTargetView> Fbo::getRenderTargetView(uint32_t rtIndex) const
{
    auto& rt = mColorAttachments[rtIndex];
    if (rt.pTexture)
    {
        return rt.pTexture->getRTV(rt.mipLevel, rt.firstArraySlice, rt.arraySize);
    }
    else
    {
        if (!rt.pNullView)
        {
            // TODO: mColorAttachments doesn't contain enough information to fully determine the view dimension. Assume 2D for now.
            auto dimension = rt.arraySize > 1 ? RenderTargetView::Dimension::Texture2DArray : RenderTargetView::Dimension::Texture2D;
            rt.pNullView = RenderTargetView::create(mpDevice, dimension);
        }
        return static_ref_cast<RenderTargetView>(rt.pNullView);
    }
}

ref<DepthStencilView> Fbo::getDepthStencilView() const
{
    if (mDepthStencil.pTexture)
    {
        return mDepthStencil.pTexture->getDSV(mDepthStencil.mipLevel, mDepthStencil.firstArraySlice, mDepthStencil.arraySize);
    }
    else
    {
        if (!mDepthStencil.pNullView)
        {
            // TODO: mDepthStencil doesn't contain enough information to fully determine the view dimension.  Assume 2D for now.
            auto dimension =
                mDepthStencil.arraySize > 1 ? DepthStencilView::Dimension::Texture2DArray : DepthStencilView::Dimension::Texture2D;
            mDepthStencil.pNullView = DepthStencilView::create(mpDevice, dimension);
        }
        return static_ref_cast<DepthStencilView>(mDepthStencil.pNullView);
    }
}

void Fbo::attachDepthStencilTarget(const ref<Texture>& pDepthStencil, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
{
    bool changed = (mDepthStencil.pTexture != pDepthStencil);
    changed |= (mDepthStencil.mipLevel != mipLevel);
    changed |= (mDepthStencil.firstArraySlice != firstArraySlice);
    changed |= (mDepthStencil.arraySize != arraySize);
    if (!changed)
        return;

    checkAttachArguments(pDepthStencil.get(), mipLevel, firstArraySlice, arraySize, true);

    mpDesc = nullptr;
    mDepthStencil.pTexture = pDepthStencil;
    mDepthStencil.mipLevel = mipLevel;
    mDepthStencil.firstArraySlice = firstArraySlice;
    mDepthStencil.arraySize = arraySize;
    bool allowUav = false;
    if (pDepthStencil)
    {
        allowUav = ((pDepthStencil->getBindFlags() & ResourceBindFlags::UnorderedAccess) != ResourceBindFlags::None);
    }

    mTempDesc.setDepthStencilTarget(pDepthStencil ? pDepthStencil->getFormat() : ResourceFormat::Unknown, allowUav);
    applyDepthAttachment();
}

void Fbo::attachColorTarget(const ref<Texture>& pTexture, uint32_t rtIndex, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
{
    FALCOR_CHECK(
        rtIndex < mColorAttachments.size(),
        "'rtIndex' ({}) is out of range. Only {} color targets are available.",
        rtIndex,
        mColorAttachments.size()
    );

    bool changed = (mColorAttachments[rtIndex].pTexture != pTexture);
    changed |= (mColorAttachments[rtIndex].mipLevel != mipLevel);
    changed |= (mColorAttachments[rtIndex].firstArraySlice != firstArraySlice);
    changed |= (mColorAttachments[rtIndex].arraySize != arraySize);
    if (!changed)
        return;

    checkAttachArguments(pTexture.get(), mipLevel, firstArraySlice, arraySize, false);

    mpDesc = nullptr;
    mColorAttachments[rtIndex].pTexture = pTexture;
    mColorAttachments[rtIndex].mipLevel = mipLevel;
    mColorAttachments[rtIndex].firstArraySlice = firstArraySlice;
    mColorAttachments[rtIndex].arraySize = arraySize;
    bool allowUav = false;
    if (pTexture)
    {
        allowUav = ((pTexture->getBindFlags() & ResourceBindFlags::UnorderedAccess) != ResourceBindFlags::None);
    }

    mTempDesc.setColorTarget(rtIndex, pTexture ? pTexture->getFormat() : ResourceFormat::Unknown, allowUav);
    applyColorAttachment(rtIndex);
}

void Fbo::validateAttachment(const Attachment& attachment) const
{
    const Texture* pTexture = attachment.pTexture.get();
    if (pTexture)
    {
        // Calculate size
        if (mWidth == uint32_t(-1))
        {
            // First attachment in the FBO
            mTempDesc.setSampleCount(pTexture->getSampleCount());
            mIsLayered = (attachment.arraySize > 1);
        }

        mWidth = std::min(mWidth, pTexture->getWidth(attachment.mipLevel));
        mHeight = std::min(mHeight, pTexture->getHeight(attachment.mipLevel));
        mDepth = std::min(mDepth, pTexture->getDepth(attachment.mipLevel));

        {
            if ((pTexture->getSampleCount() > mTempDesc.getSampleCount()) && isDepthStencilFormat(pTexture->getFormat()))
            {
                // We're using target-independent raster (more depth samples than color samples).  This is OK.
                mTempDesc.setSampleCount(pTexture->getSampleCount());
            }

            if (mTempDesc.getSampleCount() != pTexture->getSampleCount())
            {
                FALCOR_THROW("Error when validating FBO. Different sample counts in attachments.");
            }

            if (mIsLayered != (attachment.arraySize > 1))
            {
                FALCOR_THROW("Error when validating FBO. Can't bind both layered and non-layered textures.");
            }
        }
    }
}

void Fbo::calcAndValidateProperties() const
{
    mWidth = (uint32_t)-1;
    mHeight = (uint32_t)-1;
    mDepth = (uint32_t)-1;
    mTempDesc.setSampleCount(uint32_t(-1));
    mIsLayered = false;

    // Validate color attachments.
    for (const auto& attachment : mColorAttachments)
        validateAttachment(attachment);

    // Validate depth attachment.
    validateAttachment(mDepthStencil);

    // In case there are sample positions, make sure they are valid.
    if (mSamplePositions.size())
    {
        uint32_t expectedCount = mSamplePositionsPixelCount * mTempDesc.getSampleCount();
        if (expectedCount != mSamplePositions.size())
        {
            FALCOR_THROW("Error when validating FBO. The sample positions array size has the wrong size.");
        }
    }

    // The GraphicsState class relies on stable pointers of the descriptor to
    // walk the tree of cached state objects. Store the desc in a global set
    // and get a pointer to it.
    mpDesc = &(*(getGlobalDescCache().insert(mTempDesc).first));
}

ref<Texture> Fbo::getColorTexture(uint32_t index) const
{
    FALCOR_CHECK(
        index < mColorAttachments.size(),
        "'index' ({}) is out of range. Only {} color slots are available.",
        index,
        mColorAttachments.size()
    );
    return mColorAttachments[index].pTexture;
}

const ref<Texture>& Fbo::getDepthStencilTexture() const
{
    return mDepthStencil.pTexture;
}

void Fbo::finalize() const
{
    if (mpDesc == nullptr)
    {
        calcAndValidateProperties();
        initFramebuffer();
    }
}

void Fbo::setSamplePositions(uint32_t samplesPerPixel, uint32_t pixelCount, const SamplePosition positions[])
{
    if (positions)
    {
        mSamplePositions = std::vector<SamplePosition>(positions, positions + (samplesPerPixel * pixelCount));
        mSamplePositionsPixelCount = pixelCount;
    }
    else
    {
        mSamplePositionsPixelCount = 0;
        mSamplePositions.clear();
    }
}

ref<Fbo> Fbo::create2D(
    ref<Device> pDevice,
    uint32_t width,
    uint32_t height,
    const Fbo::Desc& fboDesc,
    uint32_t arraySize,
    uint32_t mipLevels
)
{
    uint32_t sampleCount = fboDesc.getSampleCount();

    FALCOR_CHECK(width > 0, "'width' must not be zero.");
    FALCOR_CHECK(height > 0, "'height' must not be zero.");
    FALCOR_CHECK(arraySize > 0, "'arraySize' must not be zero.");
    FALCOR_CHECK(mipLevels > 0, "'mipLevels' must not be zero.");
    FALCOR_CHECK(sampleCount == 1 || mipLevels == 1, "Cannot create multi-sampled texture with more than one mip-level.");

    ref<Fbo> pFbo = create(pDevice);

    // Create the color targets
    for (uint32_t i = 0; i < Fbo::getMaxColorTargetCount(); i++)
    {
        if (fboDesc.getColorTargetFormat(i) != ResourceFormat::Unknown)
        {
            ResourceBindFlags flags = getBindFlags(false, fboDesc.isColorTargetUav(i));
            ref<Texture> pTex =
                createTexture2D(pDevice, width, height, fboDesc.getColorTargetFormat(i), sampleCount, arraySize, mipLevels, flags);
            pFbo->attachColorTarget(pTex, i, 0, 0, kAttachEntireMipLevel);
        }
    }

    if (fboDesc.getDepthStencilFormat() != ResourceFormat::Unknown)
    {
        ResourceBindFlags flags = getBindFlags(true, fboDesc.isDepthStencilUav());
        ref<Texture> pDepth =
            createTexture2D(pDevice, width, height, fboDesc.getDepthStencilFormat(), sampleCount, arraySize, mipLevels, flags);
        pFbo->attachDepthStencilTarget(pDepth, 0, 0, kAttachEntireMipLevel);
    }

    return pFbo;
}

ref<Fbo> Fbo::createCubemap(
    ref<Device> pDevice,
    uint32_t width,
    uint32_t height,
    const Desc& fboDesc,
    uint32_t arraySize,
    uint32_t mipLevels
)
{
    FALCOR_CHECK(width > 0, "'width' must not be zero.");
    FALCOR_CHECK(height > 0, "'height' must not be zero.");
    FALCOR_CHECK(arraySize > 0, "'arraySize' must not be zero.");
    FALCOR_CHECK(mipLevels > 0, "'mipLevels' must not be zero.");
    FALCOR_CHECK(fboDesc.getSampleCount() == 1, "Cannot create multi-sampled cube map.");

    ref<Fbo> pFbo = create(pDevice);

    // Create the color targets
    for (uint32_t i = 0; i < getMaxColorTargetCount(); i++)
    {
        ResourceBindFlags flags = getBindFlags(false, fboDesc.isColorTargetUav(i));
        auto pTex = pDevice->createTextureCube(width, height, fboDesc.getColorTargetFormat(i), arraySize, mipLevels, nullptr, flags);
        pFbo->attachColorTarget(pTex, i, 0, kAttachEntireMipLevel);
    }

    if (fboDesc.getDepthStencilFormat() != ResourceFormat::Unknown)
    {
        ResourceBindFlags flags = getBindFlags(true, fboDesc.isDepthStencilUav());
        auto pDepth = pDevice->createTextureCube(width, height, fboDesc.getDepthStencilFormat(), arraySize, mipLevels, nullptr, flags);
        pFbo->attachDepthStencilTarget(pDepth, 0, kAttachEntireMipLevel);
    }

    return pFbo;
}

ref<Fbo> Fbo::create2D(ref<Device> pDevice, uint32_t width, uint32_t height, ResourceFormat color, ResourceFormat depth)
{
    Desc d;
    d.setColorTarget(0, color).setDepthStencilTarget(depth);
    return create2D(pDevice, width, height, d);
}

void Fbo::breakStrongReferenceToDevice()
{
    mpDevice.breakStrongReference();
}

FALCOR_SCRIPT_BINDING(Fbo)
{
    pybind11::class_<Fbo, ref<Fbo>>(m, "Fbo");
}
} // namespace Falcor
