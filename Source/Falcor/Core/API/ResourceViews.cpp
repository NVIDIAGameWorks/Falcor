/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "ResourceViews.h"

namespace Falcor
{
    static NullResourceViews gNullViews;
    Texture::SharedPtr getEmptyTexture();

    void createNullViews()
    {
        gNullViews.srv = ShaderResourceView::create(getEmptyTexture(), 0, 1, 0, 1);
        gNullViews.dsv = DepthStencilView::create(getEmptyTexture(), 0, 0, 1);
        gNullViews.uav = UnorderedAccessView::create(getEmptyTexture(), 0, 0, 1);
        gNullViews.rtv = RenderTargetView::create(getEmptyTexture(), 0, 0, 1);
        gNullViews.cbv = ConstantBufferView::create(Buffer::SharedPtr());
    }

    void releaseNullViews()
    {
        gNullViews = {};
    }

    ShaderResourceView::SharedPtr  ShaderResourceView::getNullView()  { return gNullViews.srv; }
    DepthStencilView::SharedPtr    DepthStencilView::getNullView()    { return gNullViews.dsv; }
    UnorderedAccessView::SharedPtr UnorderedAccessView::getNullView() { return gNullViews.uav; }
    RenderTargetView::SharedPtr    RenderTargetView::getNullView()    { return gNullViews.rtv;}
    ConstantBufferView::SharedPtr  ConstantBufferView::getNullView()  { return gNullViews.cbv;}

    SCRIPT_BINDING(ResourceView)
    {
        pybind11::class_<ShaderResourceView, ShaderResourceView::SharedPtr>(m, "ShaderResourceView");
        pybind11::class_<RenderTargetView, RenderTargetView::SharedPtr>(m, "RenderTargetView");
        pybind11::class_<UnorderedAccessView, UnorderedAccessView::SharedPtr>(m, "UnorderedAccessView");
        pybind11::class_<ConstantBufferView, ConstantBufferView::SharedPtr>(m, "ConstantBufferView");
        pybind11::class_<DepthStencilView, DepthStencilView::SharedPtr>(m, "DepthStencilView");
    }
}
