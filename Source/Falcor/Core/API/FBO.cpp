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
#include "FBO.h"
#include "Core/Errors.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    namespace
    {
        void checkAttachArguments(const Texture* pTexture, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize, bool isDepthAttachment)
        {
            if (pTexture == nullptr) return;

            checkArgument(mipLevel < pTexture->getMipCount(), "'mipLevel' ({}) is out of bounds.", mipLevel);

            if (arraySize != Fbo::kAttachEntireMipLevel)
            {
                checkArgument(arraySize != 0, "'arraySize' must not be zero.");
                if (pTexture->getType() == Texture::Type::Texture3D)
                {
                    checkArgument(arraySize + firstArraySlice <= pTexture->getDepth(), "'firstArraySlice' ({}) and 'arraySize' ({}) request depth index that is out of bounds.", firstArraySlice, arraySize);
                }
                else
                {
                    checkArgument(arraySize + firstArraySlice <= pTexture->getArraySize(), "'frstArraySlice' ({}) and 'arraySize' ({}) request array index that is out of bounds.", firstArraySlice, arraySize);
                }
            }

            if (isDepthAttachment)
            {
                checkArgument(isDepthStencilFormat(pTexture->getFormat()), "Depth-stencil texture must have a depth-stencil format.");
                checkArgument(is_set(pTexture->getBindFlags(), Texture::BindFlags::DepthStencil), "Depth-stencil texture must have the DepthStencil bind flag.");
            }
            else
            {
                checkArgument(!isDepthStencilFormat(pTexture->getFormat()), "Color texture must not have a depth-stencil format.");
                checkArgument(is_set(pTexture->getBindFlags(), Texture::BindFlags::RenderTarget), "Color texture must have the RenderTarget bind flag.");
            }
        }

        Texture::SharedPtr createTexture2D(uint32_t w, uint32_t h, ResourceFormat format, uint32_t sampleCount, uint32_t arraySize, uint32_t mipLevels, Texture::BindFlags flags)
        {
            if (format == ResourceFormat::Unknown)
            {
                throw RuntimeError("Can't create Texture2D with an unknown resource format.");
            }

            if (sampleCount > 1)
            {
                return Texture::create2DMS(w, h, format, sampleCount, arraySize, flags);
            }
            else
            {
                return Texture::create2D(w, h, format, arraySize, mipLevels, nullptr, flags);
            }
        }

        Texture::BindFlags getBindFlags(bool isDepth, bool allowUav)
        {
            Texture::BindFlags flags = Texture::BindFlags::ShaderResource;
            flags |= isDepth ? Texture::BindFlags::DepthStencil : Texture::BindFlags::RenderTarget;

            if (allowUav)
            {
                flags |= Texture::BindFlags::UnorderedAccess;
            }
            return flags;
        }

    };

    std::unordered_set<Fbo::Desc, Fbo::DescHash> Fbo::sDescs;

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
        if (mColorTargets.size() != other.mColorTargets.size()) return false;

        for (size_t i = 0; i < mColorTargets.size(); i++)
        {
            if (mColorTargets[i] != other.mColorTargets[i]) return false;
        }
        if (mDepthStencilTarget != other.mDepthStencilTarget) return false;
        if (mSampleCount != other.mSampleCount) return false;

        return true;
    }

    Fbo::Desc::Desc()
    {
        mColorTargets.resize(Fbo::getMaxColorTargetCount());
    }

    Fbo::SharedPtr Fbo::create()
    {
        return SharedPtr(new Fbo());
    }

    Fbo::SharedPtr Fbo::create(const std::vector<Texture::SharedPtr>& colors, const Texture::SharedPtr& pDepth)
    {
        auto pFbo = create();
        for (uint32_t i = 0 ; i < colors.size() ; i++)
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

    Fbo::SharedPtr Fbo::getDefault()
    {
        static Fbo::SharedPtr pDefault;
        if (pDefault == nullptr)
        {
            pDefault = Fbo::SharedPtr(new Fbo());
        }
        return pDefault;
    }

    void Fbo::attachDepthStencilTarget(const Texture::SharedPtr& pDepthStencil, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        bool changed = (mDepthStencil.pTexture != pDepthStencil);
        changed |= (mDepthStencil.mipLevel != mipLevel);
        changed |= (mDepthStencil.firstArraySlice != firstArraySlice);
        changed |= (mDepthStencil.arraySize != arraySize);
        if (!changed) return;

        checkAttachArguments(pDepthStencil.get(), mipLevel, firstArraySlice, arraySize, true);

        mpDesc = nullptr;
        mDepthStencil.pTexture = pDepthStencil;
        mDepthStencil.mipLevel = mipLevel;
        mDepthStencil.firstArraySlice = firstArraySlice;
        mDepthStencil.arraySize = arraySize;
        bool allowUav = false;
        if (pDepthStencil)
        {
            allowUav = ((pDepthStencil->getBindFlags() & Texture::BindFlags::UnorderedAccess) != Texture::BindFlags::None);
        }

        mTempDesc.setDepthStencilTarget(pDepthStencil ? pDepthStencil->getFormat() : ResourceFormat::Unknown, allowUav);
        applyDepthAttachment();
    }

    void Fbo::attachColorTarget(const Texture::SharedPtr& pTexture, uint32_t rtIndex, uint32_t mipLevel, uint32_t firstArraySlice, uint32_t arraySize)
    {
        checkArgument(rtIndex < mColorAttachments.size(), "'rtIndex' ({}) is out of range. Only {} color targets are available.", rtIndex, mColorAttachments.size());

        bool changed = (mColorAttachments[rtIndex].pTexture != pTexture);
        changed |= (mColorAttachments[rtIndex].mipLevel != mipLevel);
        changed |= (mColorAttachments[rtIndex].firstArraySlice != firstArraySlice);
        changed |= (mColorAttachments[rtIndex].arraySize != arraySize);
        if (!changed) return;

        checkAttachArguments(pTexture.get(), mipLevel, firstArraySlice, arraySize, false);

        mpDesc = nullptr;
        mColorAttachments[rtIndex].pTexture = pTexture;
        mColorAttachments[rtIndex].mipLevel = mipLevel;
        mColorAttachments[rtIndex].firstArraySlice = firstArraySlice;
        mColorAttachments[rtIndex].arraySize = arraySize;
        bool allowUav = false;
        if (pTexture)
        {
            allowUav = ((pTexture->getBindFlags() & Texture::BindFlags::UnorderedAccess) != Texture::BindFlags::None);
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
                if ( (pTexture->getSampleCount() > mTempDesc.getSampleCount()) && isDepthStencilFormat(pTexture->getFormat()) )
                {
                    // We're using target-independent raster (more depth samples than color samples).  This is OK.
                    mTempDesc.setSampleCount(pTexture->getSampleCount());
                }

                if (mTempDesc.getSampleCount() != pTexture->getSampleCount())
                {
                    throw RuntimeError("Error when validating FBO. Different sample counts in attachments.");
                }


                if (mIsLayered != (attachment.arraySize > 1))
                {
                    throw RuntimeError("Error when validating FBO. Can't bind both layered and non-layered textures.");
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
        for (const auto& attachment : mColorAttachments) validateAttachment(attachment);

        // Validate depth attachment.
        validateAttachment(mDepthStencil);

        // In case there are sample positions, make sure they are valid.
        if (mSamplePositions.size())
        {
            uint32_t expectedCount = mSamplePositionsPixelCount * mTempDesc.getSampleCount();
            if (expectedCount != mSamplePositions.size())
            {
                throw RuntimeError("Error when validating FBO. The sample positions array size has the wrong size.");
            }
        }

        // Insert the attachment into the static array and initialize the address.
        mpDesc = &(*(sDescs.insert(mTempDesc).first));
    }

    Texture::SharedPtr Fbo::getColorTexture(uint32_t index) const
    {
        checkArgument(index < mColorAttachments.size(), "'index' ({}) is out of range. Only {} color slots are available.", index, mColorAttachments.size());
        return mColorAttachments[index].pTexture;
    }

    const Texture::SharedPtr& Fbo::getDepthStencilTexture() const
    {
        return mDepthStencil.pTexture;
    }

    void Fbo::finalize() const
    {
        if (mpDesc == nullptr)
        {
            calcAndValidateProperties();
            initApiHandle();
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

    Fbo::SharedPtr Fbo::create2D(uint32_t width, uint32_t height, const Fbo::Desc& fboDesc, uint32_t arraySize, uint32_t mipLevels)
    {
        uint32_t sampleCount = fboDesc.getSampleCount();

        checkArgument(width > 0, "'width' must not be zero.");
        checkArgument(height > 0, "'height' must not be zero.");
        checkArgument(arraySize > 0, "'arraySize' must not be zero.");
        checkArgument(mipLevels > 0, "'mipLevels' must not be zero.");
        checkArgument(sampleCount == 1 || mipLevels == 1, "Cannot create multi-sampled texture with more than one mip-level.");

        Fbo::SharedPtr pFbo = create();

        // Create the color targets
        for (uint32_t i = 0; i < Fbo::getMaxColorTargetCount(); i++)
        {
            if (fboDesc.getColorTargetFormat(i) != ResourceFormat::Unknown)
            {
                Texture::BindFlags flags = getBindFlags(false, fboDesc.isColorTargetUav(i));
                Texture::SharedPtr pTex = createTexture2D(width, height, fboDesc.getColorTargetFormat(i), sampleCount, arraySize, mipLevels, flags);
                pFbo->attachColorTarget(pTex, i, 0, 0, kAttachEntireMipLevel);
            }
        }

        if (fboDesc.getDepthStencilFormat() != ResourceFormat::Unknown)
        {
            Texture::BindFlags flags = getBindFlags(true, fboDesc.isDepthStencilUav());
            Texture::SharedPtr pDepth = createTexture2D(width, height, fboDesc.getDepthStencilFormat(), sampleCount, arraySize, mipLevels, flags);
            pFbo->attachDepthStencilTarget(pDepth, 0, 0, kAttachEntireMipLevel);
        }

        return pFbo;
    }

    Fbo::SharedPtr Fbo::createCubemap(uint32_t width, uint32_t height, const Desc& fboDesc, uint32_t arraySize, uint32_t mipLevels)
    {
        checkArgument(width > 0, "'width' must not be zero.");
        checkArgument(height > 0, "'height' must not be zero.");
        checkArgument(arraySize > 0, "'arraySize' must not be zero.");
        checkArgument(mipLevels > 0, "'mipLevels' must not be zero.");
        checkArgument(fboDesc.getSampleCount() == 1, "Cannot create multi-sampled cube map.");

        Fbo::SharedPtr pFbo = create();

        // Create the color targets
        for (uint32_t i = 0; i < getMaxColorTargetCount(); i++)
        {
            Texture::BindFlags flags = getBindFlags(false, fboDesc.isColorTargetUav(i));
            auto pTex = Texture::createCube(width, height, fboDesc.getColorTargetFormat(i), arraySize, mipLevels, nullptr, flags);
            pFbo->attachColorTarget(pTex, i, 0, kAttachEntireMipLevel);
        }

        if (fboDesc.getDepthStencilFormat() != ResourceFormat::Unknown)
        {
            Texture::BindFlags flags = getBindFlags(true, fboDesc.isDepthStencilUav());
            auto pDepth = Texture::createCube(width, height, fboDesc.getDepthStencilFormat(), arraySize, mipLevels, nullptr, flags);
            pFbo->attachDepthStencilTarget(pDepth, 0, kAttachEntireMipLevel);
        }

        return pFbo;
    }

    Fbo::SharedPtr Fbo::create2D(uint32_t width, uint32_t height, ResourceFormat color, ResourceFormat depth)
    {
        Desc d;
        d.setColorTarget(0, color).setDepthStencilTarget(depth);
        return create2D(width, height, d);
    }

    FALCOR_SCRIPT_BINDING(Fbo)
    {
        pybind11::class_<Fbo, Fbo::SharedPtr>(m, "Fbo");
    }
}
