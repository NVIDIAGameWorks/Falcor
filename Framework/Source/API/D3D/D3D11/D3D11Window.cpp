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
#include "Framework.h"
#include "Sample.h"

namespace Falcor
{
    ID3D11DevicePtr gpD3D11Device = nullptr;
    ID3D11DeviceContextPtr gpD3D11ImmediateContext = nullptr;

    ID3D11DevicePtr getD3D11Device()
    {
        return gpD3D11Device;
    }

    ID3D11DeviceContextPtr getD3D11ImmediateContext()
    {
        return gpD3D11ImmediateContext;
    }

    ID3D11ResourcePtr createDepthTexture(uint32_t width, uint32_t height, uint32_t sampleCount, ResourceFormat depthFormat)
    {
        // create the depth stencil resource and view
        D3D11_TEXTURE2D_DESC depthDesc;
        depthDesc.ArraySize = 1;
        depthDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        depthDesc.CPUAccessFlags = 0;
        depthDesc.Format = getDxgiFormat(depthFormat);
        depthDesc.Height = height;
        depthDesc.Width = width;
        depthDesc.MipLevels = 1;
        depthDesc.MiscFlags = 0;
        depthDesc.SampleDesc.Count = sampleCount;

        depthDesc.SampleDesc.Quality = 0;
        depthDesc.Usage = D3D11_USAGE_DEFAULT;
        ID3D11Texture2DPtr pDepthResource;
        d3d_call(gpD3D11Device->CreateTexture2D(&depthDesc, nullptr, &pDepthResource));

        return pDepthResource;
    }

    IUnknown* createDevice(uint32_t apiMajorVersion, uint32_t apiMinorVersion)
    {
        if(gpD3D11Device)
        {
            logError("DX11 backend doesn't support more than a single device.");
            return nullptr;
        }

        UINT flags = 0;
        if(desc.enableDebugLayer)
        {
            flags |= D3D11_CREATE_DEVICE_DEBUG;
        }

        D3D_FEATURE_LEVEL level = getD3DFeatureLevel(apiMajorVersion, apiMinorVersion);
        if(level == 0)
        {
            Logger::log(Logger::Level::Fatal, "Unsupported device feature level requested: " + std::to_string(apiMajorVersion) + "." + std::to_string(apiMinorVersion));
            return nullptr;
        }

        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags, &level, 1, D3D11_SDK_VERSION, &gpD3D11Device, nullptr, nullptr);

        if(FAILED(hr))
        {
            d3dTraceHR("Failed to create DX device", hr);
            return nullptr;
        }

        // Get the immediate context
        gpD3D11Device->GetImmediateContext(&gpD3D11ImmediateContext);

        return gpD3D11Device;
    }
}
