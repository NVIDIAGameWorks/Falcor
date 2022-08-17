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
#include "GridVolumeSampler.h"
#include "Core/Assert.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    namespace
    {
        const Gui::DropdownList kTransmittanceEstimatorList =
        {
            { (uint32_t)TransmittanceEstimator::DeltaTracking, "Delta Tracking (Global Majorant)" },
            { (uint32_t)TransmittanceEstimator::RatioTracking, "Ratio Tracking (Global Majorant)" },
            { (uint32_t)TransmittanceEstimator::RatioTrackingLocalMajorant, "Ratio Tracking (Local Majorants)" },
        };
        const Gui::DropdownList kDistanceSamplerList =
        {
            { (uint32_t)DistanceSampler::DeltaTracking, "Delta Tracking (Global Majorant)" },
            { (uint32_t)DistanceSampler::DeltaTrackingLocalMajorant, "Delta Tracking (Local Majorants)" },
        };
    }

    GridVolumeSampler::SharedPtr GridVolumeSampler::create(RenderContext* pRenderContext, Scene::SharedPtr pScene, const Options& options)
    {
        return SharedPtr(new GridVolumeSampler(pRenderContext, pScene, options));
    }

    Program::DefineList GridVolumeSampler::getDefines() const
    {
        Program::DefineList defines;
        defines.add("GRID_VOLUME_SAMPLER_USE_BRICKEDGRID", std::to_string((uint32_t)mOptions.useBrickedGrid));
        defines.add("GRID_VOLUME_SAMPLER_TRANSMITTANCE_ESTIMATOR", std::to_string((uint32_t)mOptions.transmittanceEstimator));
        defines.add("GRID_VOLUME_SAMPLER_DISTANCE_SAMPLER", std::to_string((uint32_t)mOptions.distanceSampler));
        return defines;
    }

    void GridVolumeSampler::setShaderData(const ShaderVar& var) const
    {
        FALCOR_ASSERT(var.isValid());
    }

    bool GridVolumeSampler::renderUI(Gui::Widgets& widget)
    {
        bool dirty = false;

        if (widget.checkbox("Use BrickedGrid", mOptions.useBrickedGrid))
        {
            if (!mOptions.useBrickedGrid) {
                // Switch back to modes not requiring bricked grid.
                if (requiresBrickedGrid(mOptions.transmittanceEstimator)) mOptions.transmittanceEstimator = TransmittanceEstimator::RatioTracking;
                if (requiresBrickedGrid(mOptions.distanceSampler)) mOptions.distanceSampler = DistanceSampler::DeltaTracking;
            }
            dirty = true;
        }
        if (widget.dropdown("Transmittance Estimator", kTransmittanceEstimatorList, reinterpret_cast<uint32_t&>(mOptions.transmittanceEstimator)))
        {
            // Enable bricked grid if the chosen mode requires it.
            if (requiresBrickedGrid(mOptions.transmittanceEstimator)) mOptions.useBrickedGrid = true;
            dirty = true;
        }
        if (widget.dropdown("Distance Sampler", kDistanceSamplerList, reinterpret_cast<uint32_t&>(mOptions.distanceSampler)))
        {
            // Enable bricked grid if the chosen mode requires it.
            if (requiresBrickedGrid(mOptions.distanceSampler)) mOptions.useBrickedGrid = true;
            dirty = true;
        }

        return dirty;
    }

    GridVolumeSampler::GridVolumeSampler(RenderContext* pRenderContext, Scene::SharedPtr pScene, const Options& options)
        : mpScene(pScene)
        , mOptions(options)
    {
        FALCOR_ASSERT(pScene);
    }

    FALCOR_SCRIPT_BINDING(GridVolumeSampler)
    {
        pybind11::enum_<TransmittanceEstimator> transmittanceEstimator(m, "TransmittanceEstimator");
        transmittanceEstimator.value("DeltaTracking", TransmittanceEstimator::DeltaTracking);
        transmittanceEstimator.value("RatioTracking", TransmittanceEstimator::RatioTracking);
        transmittanceEstimator.value("RatioTrackingLocalMajorant", TransmittanceEstimator::RatioTrackingLocalMajorant);

        pybind11::enum_<DistanceSampler> distanceSampler(m, "DistanceSampler");
        distanceSampler.value("DeltaTracking", DistanceSampler::DeltaTracking);
        distanceSampler.value("DeltaTrackingLocalMajorant", DistanceSampler::DeltaTrackingLocalMajorant);

        // TODO use a nested class in the bindings when supported.
        ScriptBindings::SerializableStruct<GridVolumeSampler::Options> options(m, "GridVolumeSamplerOptions");
#define field(f_) field(#f_, &GridVolumeSampler::Options::f_)
        options.field(transmittanceEstimator);
        options.field(distanceSampler);
        options.field(useBrickedGrid);
#undef field
    }
}
