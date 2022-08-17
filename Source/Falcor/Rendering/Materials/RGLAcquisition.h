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
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "RenderGraph/BasePasses/ComputePass.h"
#include "Scene/Scene.h"
#include "Scene/Material/RGLFile.h"
#include <memory>

namespace Falcor
{
    /** This class allows taking a virtual measurement of a BRDF
        and converting it into the parametrization proposed by
        Dupuy & Jakob,  "An Adaptive Parameterization for Efficient
        Material Acquisition and Rendering".
    */
    class FALCOR_API RGLAcquisition
    {
    public:
        using SharedPtr = std::shared_ptr<RGLAcquisition>;

        static SharedPtr create(RenderContext* pRenderContext, const Scene::SharedPtr& pScene);

        void acquireIsotropic(RenderContext* pRenderContext, const MaterialID materialID);
        RGLFile toRGLFile();

    private:
        RGLAcquisition(RenderContext* pRenderContext, const Scene::SharedPtr& pScene);

        Scene::SharedPtr mpScene;
        ComputePass::SharedPtr mpRetroReflectionPass;
        ComputePass::SharedPtr mpBuildKernelPass;
        ComputePass::SharedPtr mpPowerIterationPass;
        ComputePass::SharedPtr mpIntegrateSigmaPass;
        ComputePass::SharedPtr mpSumSigmaPass;
        ComputePass::SharedPtr mpComputeThetaPass;
        ComputePass::SharedPtr mpComputeVNDFPass;
        ComputePass::SharedPtr mpAcquireBRDFPass;

        Buffer::SharedPtr mpNDFDirectionsBuffer;            ///< Stores hemispherical directions of entries in the NDF table.
        Buffer::SharedPtr mpRetroBuffer;                    ///< 2D table storing measured retroreflecton of BRDF.
        Buffer::SharedPtr mpNDFKernelBuffer;                ///< Stores kernel matrix of Fredholm problem for retrieving NDF.
        Buffer::SharedPtr mpNDFBuffer;                      ///< 2D table storing the retrieved NDF.
        Buffer::SharedPtr mpNDFBufferTmp;
        Buffer::SharedPtr mpSigmaBuffer;                    ///< 2D table of projected microfacet area, integrated numerically.
        Buffer::SharedPtr mpThetaBuffer;                    ///< 1D tables storing angles at which measurements are taken.
        Buffer::SharedPtr mpPhiBuffer;
        Buffer::SharedPtr mpVNDFBuffer;                     ///< 4D table (over wi x wo domains) containing the visible distribution of normals.
        Buffer::SharedPtr mpVNDFMargBuffer;                 ///< Auxiliary buffers for sampling the VNDF.
        Buffer::SharedPtr mpVNDFCondBuffer;
        Buffer::SharedPtr mpLumiBuffer;                     ///< 4D table of measured luminance and RGB reflectance of RGB.
        Buffer::SharedPtr mpRGBBuffer;
    };
}
