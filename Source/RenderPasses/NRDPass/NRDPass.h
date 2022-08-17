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
#pragma once

#include <Falcor.h>
#include "Core/API/Shared/D3D12DescriptorSet.h"
#include "Core/API/Shared/D3D12RootSignature.h"
#include "RenderGraph/RenderPassHelpers.h"

#include <NRD.h>

using namespace Falcor;

class NRDPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<NRDPass>;

    static const Info kInfo;

    enum class DenoisingMethod : uint32_t
    {
        RelaxDiffuseSpecular,
        RelaxDiffuse,
        ReblurDiffuseSpecular,
        SpecularReflectionMv,
        SpecularDeltaMv
    };

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;

private:
    NRDPass(const Dictionary& dict);

    Scene::SharedPtr mpScene;
    uint2 mScreenSize{};
    uint32_t mFrameIndex = 0;
    RenderPassHelpers::IOSize  mOutputSizeSelection = RenderPassHelpers::IOSize::Default; ///< Selected output size.

    void reinit();
    void createPipelines();
    void createResources();
    void executeInternal(RenderContext* pRenderContext, const RenderData& renderData);
    void dispatch(RenderContext* pRenderContext, const RenderData& renderData, const nrd::DispatchDesc& dispatchDesc);

    nrd::Denoiser* mpDenoiser = nullptr;

    bool mEnabled = true;
    DenoisingMethod mDenoisingMethod = DenoisingMethod::RelaxDiffuseSpecular;
    bool mRecreateDenoiser = false;
    bool mWorldSpaceMotion = true;
    float mMaxIntensity = 1000.f;
    float mDisocclusionThreshold = 2.f;
    nrd::CommonSettings mCommonSettings = {};
    nrd::RelaxDiffuseSpecularSettings mRelaxDiffuseSpecularSettings = {};
    nrd::RelaxDiffuseSettings mRelaxDiffuseSettings = {};
    nrd::ReblurSettings mReblurSettings = {};

    std::vector<Falcor::Sampler::SharedPtr> mpSamplers;
    std::vector<Falcor::D3D12DescriptorSet::Layout> mCBVSRVUAVdescriptorSetLayouts;
    Falcor::D3D12DescriptorSet::SharedPtr mpSamplersDescriptorSet;
    std::vector<Falcor::D3D12RootSignature::SharedPtr> mpRootSignatures;
    std::vector<ComputePass::SharedPtr> mpPasses;
    std::vector<ProgramKernels::SharedConstPtr> mpCachedProgramKernels;
    std::vector<ComputeStateObject::SharedPtr> mpCSOs;
    std::vector<Falcor::Texture::SharedPtr> mpPermanentTextures;
    std::vector<Falcor::Texture::SharedPtr> mpTransientTextures;
    Falcor::Buffer::SharedPtr mpConstantBuffer;

    rmcv::mat4 mPrevViewMatrix;
    rmcv::mat4 mPrevProjMatrix;

    // Additional classic Falcor compute pass and resources for packing radiance and hitT for NRD.
    ComputePass::SharedPtr mpPackRadiancePassRelax;
    ComputePass::SharedPtr mpPackRadiancePassReblur;
};
