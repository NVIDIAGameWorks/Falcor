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
#include "Core/API/Sampler.h"
#include "Core/API/Device.h"
#include "Core/API/GFX/GFXAPI.h"
#include "Core/Assert.h"

namespace Falcor
{
    namespace
    {
        gfx::TextureAddressingMode getGFXAddressMode(Sampler::AddressMode mode)
        {
            switch (mode)
            {
            case Sampler::AddressMode::Border:
                return gfx::TextureAddressingMode::ClampToBorder;
            case Sampler::AddressMode::Clamp:
                return gfx::TextureAddressingMode::ClampToEdge;
            case Sampler::AddressMode::Mirror:
                return gfx::TextureAddressingMode::MirrorRepeat;
            case Sampler::AddressMode::MirrorOnce:
                return gfx::TextureAddressingMode::MirrorOnce;
            case Sampler::AddressMode::Wrap:
                return gfx::TextureAddressingMode::Wrap;
            default:
                FALCOR_UNREACHABLE();
                return gfx::TextureAddressingMode::ClampToBorder;
            }
        }

        gfx::TextureFilteringMode getGFXFilter(Sampler::Filter filter)
        {
            switch (filter)
            {
            case Sampler::Filter::Linear:
                return gfx::TextureFilteringMode::Linear;
            case Sampler::Filter::Point:
                return gfx::TextureFilteringMode::Point;
            default:
                FALCOR_UNREACHABLE();
                return gfx::TextureFilteringMode::Point;
            }
        }

        gfx::TextureReductionOp getGFXReductionMode(Sampler::ReductionMode mode)
        {
            switch (mode)
            {
            case Falcor::Sampler::ReductionMode::Standard:
                return gfx::TextureReductionOp::Average;
            case Falcor::Sampler::ReductionMode::Comparison:
                return gfx::TextureReductionOp::Comparison;
            case Falcor::Sampler::ReductionMode::Min:
                return gfx::TextureReductionOp::Minimum;
            case Falcor::Sampler::ReductionMode::Max:
                return gfx::TextureReductionOp::Maximum;
            default:
                return gfx::TextureReductionOp::Average;
                break;
            }
        }
    }

    gfx::ComparisonFunc getGFXComparisonFunc(ComparisonFunc func);

    uint32_t Sampler::getApiMaxAnisotropy()
    {
        return 16;
    }

    Sampler::SharedPtr Sampler::create(const Desc& desc)
    {
        gfx::ISamplerState::Desc gfxDesc = {};
        gfxDesc.addressU = getGFXAddressMode(desc.mModeU);
        gfxDesc.addressV = getGFXAddressMode(desc.mModeV);
        gfxDesc.addressW = getGFXAddressMode(desc.mModeW);

        static_assert(sizeof(gfxDesc.borderColor) == sizeof(desc.mBorderColor));
        memcpy(gfxDesc.borderColor, &desc.mBorderColor, sizeof(desc.mBorderColor));

        gfxDesc.comparisonFunc = getGFXComparisonFunc(desc.mComparisonMode);
        gfxDesc.magFilter = getGFXFilter(desc.mMagFilter);
        gfxDesc.maxAnisotropy = desc.mMaxAnisotropy;
        gfxDesc.maxLOD = desc.mMaxLod;
        gfxDesc.minFilter = getGFXFilter(desc.mMinFilter);
        gfxDesc.minLOD = desc.mMinLod;
        gfxDesc.mipFilter = getGFXFilter(desc.mMipFilter);
        gfxDesc.mipLODBias = desc.mLodBias;
        gfxDesc.reductionOp = (desc.mComparisonMode != Sampler::ComparisonMode::Disabled) ? gfx::TextureReductionOp::Comparison : getGFXReductionMode(desc.mReductionMode);

        Sampler::SharedPtr result = Sampler::SharedPtr(new Sampler(desc));
        FALCOR_GFX_CALL(gpDevice->getApiHandle()->createSamplerState(gfxDesc, result->mApiHandle.writeRef()));
        return result;
    }

    D3D12DescriptorCpuHandle Sampler::getD3D12CpuHeapHandle() const
    {
#if FALCOR_HAS_D3D12
        gfx::InteropHandle handle = {};
        FALCOR_GFX_CALL(mApiHandle->getNativeHandle(&handle));
        FALCOR_ASSERT(handle.api == gfx::InteropHandleAPI::D3D12CpuDescriptorHandle);

        D3D12DescriptorCpuHandle resultHandle;
        resultHandle.ptr = handle.handleValue;
        return resultHandle;
#else
        return nullptr;
#endif
    }
}
