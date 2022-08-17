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
#include "ResourceViews.h"
#include "Device.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    namespace
    {
        struct NullResourceViews
        {
            std::array<ShaderResourceView::SharedPtr, (size_t)ShaderResourceView::Dimension::Count> srv;
            std::array<UnorderedAccessView::SharedPtr, (size_t)UnorderedAccessView::Dimension::Count> uav;
            std::array<DepthStencilView::SharedPtr, (size_t)DepthStencilView::Dimension::Count> dsv;
            std::array<RenderTargetView::SharedPtr, (size_t)RenderTargetView::Dimension::Count> rtv;
            ConstantBufferView::SharedPtr cbv;
        };

        NullResourceViews gNullViews;
    }

    void createNullViews()
    {
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::Buffer] = ShaderResourceView::create(ShaderResourceView::Dimension::Buffer);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::Texture1D] = ShaderResourceView::create(ShaderResourceView::Dimension::Texture1D);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::Texture1DArray] = ShaderResourceView::create(ShaderResourceView::Dimension::Texture1DArray);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::Texture2D] = ShaderResourceView::create(ShaderResourceView::Dimension::Texture2D);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::Texture2DArray] = ShaderResourceView::create(ShaderResourceView::Dimension::Texture2DArray);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::Texture2DMS] = ShaderResourceView::create(ShaderResourceView::Dimension::Texture2DMS);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::Texture2DMSArray] = ShaderResourceView::create(ShaderResourceView::Dimension::Texture2DMSArray);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::Texture3D] = ShaderResourceView::create(ShaderResourceView::Dimension::Texture3D);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::TextureCube] = ShaderResourceView::create(ShaderResourceView::Dimension::TextureCube);
        gNullViews.srv[(size_t)ShaderResourceView::Dimension::TextureCubeArray] = ShaderResourceView::create(ShaderResourceView::Dimension::TextureCubeArray);

        if (gpDevice->isFeatureSupported(Device::SupportedFeatures::Raytracing))
        {
            gNullViews.srv[(size_t)ShaderResourceView::Dimension::AccelerationStructure] = ShaderResourceView::create(ShaderResourceView::Dimension::AccelerationStructure);
        }

        gNullViews.uav[(size_t)UnorderedAccessView::Dimension::Buffer] = UnorderedAccessView::create(UnorderedAccessView::Dimension::Buffer);
        gNullViews.uav[(size_t)UnorderedAccessView::Dimension::Texture1D] = UnorderedAccessView::create(UnorderedAccessView::Dimension::Texture1D);
        gNullViews.uav[(size_t)UnorderedAccessView::Dimension::Texture1DArray] = UnorderedAccessView::create(UnorderedAccessView::Dimension::Texture1DArray);
        gNullViews.uav[(size_t)UnorderedAccessView::Dimension::Texture2D] = UnorderedAccessView::create(UnorderedAccessView::Dimension::Texture2D);
        gNullViews.uav[(size_t)UnorderedAccessView::Dimension::Texture2DArray] = UnorderedAccessView::create(UnorderedAccessView::Dimension::Texture2DArray);
        gNullViews.uav[(size_t)UnorderedAccessView::Dimension::Texture3D] = UnorderedAccessView::create(UnorderedAccessView::Dimension::Texture3D);

        gNullViews.dsv[(size_t)DepthStencilView::Dimension::Texture1D] = DepthStencilView::create(DepthStencilView::Dimension::Texture1D);
        gNullViews.dsv[(size_t)DepthStencilView::Dimension::Texture1DArray] = DepthStencilView::create(DepthStencilView::Dimension::Texture1DArray);
        gNullViews.dsv[(size_t)DepthStencilView::Dimension::Texture2D] = DepthStencilView::create(DepthStencilView::Dimension::Texture2D);
        gNullViews.dsv[(size_t)DepthStencilView::Dimension::Texture2DArray] = DepthStencilView::create(DepthStencilView::Dimension::Texture2DArray);
        gNullViews.dsv[(size_t)DepthStencilView::Dimension::Texture2DMS] = DepthStencilView::create(DepthStencilView::Dimension::Texture2DMS);
        gNullViews.dsv[(size_t)DepthStencilView::Dimension::Texture2DMSArray] = DepthStencilView::create(DepthStencilView::Dimension::Texture2DMSArray);

        gNullViews.rtv[(size_t)RenderTargetView::Dimension::Buffer] = RenderTargetView::create(RenderTargetView::Dimension::Buffer);
        gNullViews.rtv[(size_t)RenderTargetView::Dimension::Texture1D] = RenderTargetView::create(RenderTargetView::Dimension::Texture1D);
        gNullViews.rtv[(size_t)RenderTargetView::Dimension::Texture1DArray] = RenderTargetView::create(RenderTargetView::Dimension::Texture1DArray);
        gNullViews.rtv[(size_t)RenderTargetView::Dimension::Texture2D] = RenderTargetView::create(RenderTargetView::Dimension::Texture2D);
        gNullViews.rtv[(size_t)RenderTargetView::Dimension::Texture2DArray] = RenderTargetView::create(RenderTargetView::Dimension::Texture2DArray);
        gNullViews.rtv[(size_t)RenderTargetView::Dimension::Texture2DMS] = RenderTargetView::create(RenderTargetView::Dimension::Texture2DMS);
        gNullViews.rtv[(size_t)RenderTargetView::Dimension::Texture2DMSArray] = RenderTargetView::create(RenderTargetView::Dimension::Texture2DMSArray);
        gNullViews.rtv[(size_t)RenderTargetView::Dimension::Texture3D] = RenderTargetView::create(RenderTargetView::Dimension::Texture3D);

        gNullViews.cbv = ConstantBufferView::create();
    }

    void releaseNullViews()
    {
        gNullViews = {};
    }

    ShaderResourceView::SharedPtr ShaderResourceView::getNullView(ShaderResourceView::Dimension dimension)
    {
        FALCOR_ASSERT((size_t)dimension < gNullViews.srv.size() && gNullViews.srv[(size_t)dimension]);
        return gNullViews.srv[(size_t)dimension];
    }

    UnorderedAccessView::SharedPtr UnorderedAccessView::getNullView(UnorderedAccessView::Dimension dimension)
    {
        FALCOR_ASSERT((size_t)dimension < gNullViews.uav.size() && gNullViews.uav[(size_t)dimension]);
        return gNullViews.uav[(size_t)dimension];
    }

    DepthStencilView::SharedPtr DepthStencilView::getNullView(DepthStencilView::Dimension dimension)
    {
        FALCOR_ASSERT((size_t)dimension < gNullViews.dsv.size() && gNullViews.dsv[(size_t)dimension]);
        return gNullViews.dsv[(size_t)dimension];
    }

    RenderTargetView::SharedPtr RenderTargetView::getNullView(RenderTargetView::Dimension dimension)
    {
        FALCOR_ASSERT((size_t)dimension < gNullViews.rtv.size() && gNullViews.rtv[(size_t)dimension]);
        return gNullViews.rtv[(size_t)dimension];
    }

    ConstantBufferView::SharedPtr ConstantBufferView::getNullView()
    {
        return gNullViews.cbv;
    }

    FALCOR_SCRIPT_BINDING(ResourceView)
    {
        pybind11::class_<ShaderResourceView, ShaderResourceView::SharedPtr>(m, "ShaderResourceView");
        pybind11::class_<RenderTargetView, RenderTargetView::SharedPtr>(m, "RenderTargetView");
        pybind11::class_<UnorderedAccessView, UnorderedAccessView::SharedPtr>(m, "UnorderedAccessView");
        pybind11::class_<ConstantBufferView, ConstantBufferView::SharedPtr>(m, "ConstantBufferView");
        pybind11::class_<DepthStencilView, DepthStencilView::SharedPtr>(m, "DepthStencilView");
    }
}
