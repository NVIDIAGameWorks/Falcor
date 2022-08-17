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
#include "GridVolumeSamplerParams.slang"
#include "Core/Macros.h"
#include "Utils/UI/Gui.h"
#include "Scene/Scene.h"
#include <memory>

namespace Falcor
{
    class RenderContext;
    struct ShaderVar;

    /** Grid volume sampler.
        Utily class for evaluating transmittance and sampling scattering in grid volumes.
    */
    class FALCOR_API GridVolumeSampler
    {
    public:
        using SharedPtr = std::shared_ptr<GridVolumeSampler>;

        /** Grid volume sampler configuration options.
        */
        struct Options
        {
            TransmittanceEstimator transmittanceEstimator = TransmittanceEstimator::RatioTrackingLocalMajorant;
            DistanceSampler distanceSampler = DistanceSampler::DeltaTrackingLocalMajorant;
            bool useBrickedGrid = true;

            // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
            Options() {}
        };

        virtual ~GridVolumeSampler() = default;

        /** Create a new object.
            \param[in] pRenderContext A render-context that will be used for processing.
            \param[in] pScene The scene.
            \param[in] options Configuration options.
        */
        static SharedPtr create(RenderContext* pRenderContext, Scene::SharedPtr pScene, const Options& options = Options());

        /** Get a list of shader defines for using the grid volume sampler.
            \return Returns a list of defines.
        */
        Program::DefineList getDefines() const;

        /** Bind the grid volume sampler to a given shader variable.
            \param[in] var Shader variable.
        */
        void setShaderData(const ShaderVar& var) const;

        /** Render the GUI.
            \return True if options were changed, false otherwise.
        */
        bool renderUI(Gui::Widgets& widget);

        /** Returns the current configuration.
        */
        const Options& getOptions() const { return mOptions; }

    protected:
        GridVolumeSampler(RenderContext* pRenderContext, Scene::SharedPtr pScene, const Options& options);

        Scene::SharedPtr        mpScene;            ///< Scene.

        Options                 mOptions;
    };
}
