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
#include "Sampler.h"
#include "Device.h"
#include "GFXAPI.h"
#include "NativeHandleTraits.h"
#include "Core/ObjectPython.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
namespace
{
gfx::TextureAddressingMode getGFXAddressMode(TextureAddressingMode mode)
{
    switch (mode)
    {
    case TextureAddressingMode::Border:
        return gfx::TextureAddressingMode::ClampToBorder;
    case TextureAddressingMode::Clamp:
        return gfx::TextureAddressingMode::ClampToEdge;
    case TextureAddressingMode::Mirror:
        return gfx::TextureAddressingMode::MirrorRepeat;
    case TextureAddressingMode::MirrorOnce:
        return gfx::TextureAddressingMode::MirrorOnce;
    case TextureAddressingMode::Wrap:
        return gfx::TextureAddressingMode::Wrap;
    default:
        FALCOR_UNREACHABLE();
        return gfx::TextureAddressingMode::ClampToBorder;
    }
}

gfx::TextureFilteringMode getGFXFilter(TextureFilteringMode filter)
{
    switch (filter)
    {
    case TextureFilteringMode::Linear:
        return gfx::TextureFilteringMode::Linear;
    case TextureFilteringMode::Point:
        return gfx::TextureFilteringMode::Point;
    default:
        FALCOR_UNREACHABLE();
        return gfx::TextureFilteringMode::Point;
    }
}

gfx::TextureReductionOp getGFXReductionMode(TextureReductionMode mode)
{
    switch (mode)
    {
    case Falcor::TextureReductionMode::Standard:
        return gfx::TextureReductionOp::Average;
    case Falcor::TextureReductionMode::Comparison:
        return gfx::TextureReductionOp::Comparison;
    case Falcor::TextureReductionMode::Min:
        return gfx::TextureReductionOp::Minimum;
    case Falcor::TextureReductionMode::Max:
        return gfx::TextureReductionOp::Maximum;
    default:
        return gfx::TextureReductionOp::Average;
        break;
    }
}
} // namespace

gfx::ComparisonFunc getGFXComparisonFunc(ComparisonFunc func);

Sampler::Sampler(ref<Device> pDevice, const Desc& desc) : mpDevice(std::move(pDevice)), mDesc(desc)
{
    gfx::ISamplerState::Desc gfxDesc = {};
    gfxDesc.addressU = getGFXAddressMode(desc.addressModeU);
    gfxDesc.addressV = getGFXAddressMode(desc.addressModeV);
    gfxDesc.addressW = getGFXAddressMode(desc.addressModeW);

    static_assert(sizeof(gfxDesc.borderColor) == sizeof(desc.borderColor));
    std::memcpy(gfxDesc.borderColor, &desc.borderColor, sizeof(desc.borderColor));

    gfxDesc.comparisonFunc = getGFXComparisonFunc(desc.comparisonFunc);
    gfxDesc.magFilter = getGFXFilter(desc.magFilter);
    gfxDesc.maxAnisotropy = desc.maxAnisotropy;
    gfxDesc.maxLOD = desc.maxLod;
    gfxDesc.minFilter = getGFXFilter(desc.minFilter);
    gfxDesc.minLOD = desc.minLod;
    gfxDesc.mipFilter = getGFXFilter(desc.mipFilter);
    gfxDesc.mipLODBias = desc.lodBias;
    gfxDesc.reductionOp =
        (desc.comparisonFunc != ComparisonFunc::Disabled) ? gfx::TextureReductionOp::Comparison : getGFXReductionMode(desc.reductionMode);

    FALCOR_GFX_CALL(mpDevice->getGfxDevice()->createSamplerState(gfxDesc, mGfxSamplerState.writeRef()));
}

Sampler::~Sampler()
{
    mpDevice->releaseResource(mGfxSamplerState);
}

NativeHandle Sampler::getNativeHandle() const
{
    gfx::InteropHandle gfxNativeHandle = {};
    FALCOR_GFX_CALL(mGfxSamplerState->getNativeHandle(&gfxNativeHandle));
#if FALCOR_HAS_D3D12
    if (mpDevice->getType() == Device::Type::D3D12)
        return NativeHandle(D3D12_CPU_DESCRIPTOR_HANDLE{gfxNativeHandle.handleValue});
#endif
#if FALCOR_HAS_VULKAN
    if (mpDevice->getType() == Device::Type::Vulkan)
        return NativeHandle(reinterpret_cast<VkSampler>(gfxNativeHandle.handleValue));
#endif
    return {};
}

uint32_t Sampler::getApiMaxAnisotropy()
{
    return 16;
}

void Sampler::breakStrongReferenceToDevice()
{
    mpDevice.breakStrongReference();
}

FALCOR_SCRIPT_BINDING(Sampler)
{
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Types)

    pybind11::falcor_enum<TextureFilteringMode>(m, "TextureFilteringMode");
    pybind11::falcor_enum<TextureAddressingMode>(m, "TextureAddressingMode");
    pybind11::falcor_enum<TextureReductionMode>(m, "TextureReductionMode");

    pybind11::class_<Sampler, ref<Sampler>> sampler(m, "Sampler");
    sampler.def_property_readonly("mag_filter", &Sampler::getMagFilter);
    sampler.def_property_readonly("min_filter", &Sampler::getMinFilter);
    sampler.def_property_readonly("mip_filter", &Sampler::getMipFilter);
    sampler.def_property_readonly("max_anisotropy", &Sampler::getMaxAnisotropy);
    sampler.def_property_readonly("min_lod", &Sampler::getMinLod);
    sampler.def_property_readonly("max_lod", &Sampler::getMaxLod);
    sampler.def_property_readonly("lod_bias", &Sampler::getLodBias);
    sampler.def_property_readonly("comparison_func", &Sampler::getComparisonFunc);
    sampler.def_property_readonly("reduction_mode", &Sampler::getReductionMode);
    sampler.def_property_readonly("address_mode_u", &Sampler::getAddressModeU);
    sampler.def_property_readonly("address_mode_v", &Sampler::getAddressModeV);
    sampler.def_property_readonly("address_mode_w", &Sampler::getAddressModeW);
    sampler.def_property_readonly("border_color", &Sampler::getBorderColor);
}
} // namespace Falcor
