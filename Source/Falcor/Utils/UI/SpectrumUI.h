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
#include "Utils/Math/Vector.h"
#include "Utils/Color/SampledSpectrum.h"
#include "Utils/UI/Gui.h"
#include <imgui.h>
#include <string>
#include <vector>

namespace Falcor
{
    namespace
    {
        Gui::DropdownList kInterpolationDropdownList =
        {
            { (uint32_t)SpectrumInterpolation::Linear, "Linear" },
        };
    }

    /** User interface for SampledSpectrum<T>. Implemented using ImGui and Gui::Widgets. Can be called as widget.spectrum(...) or renderSpectrumUI(...).
    */
    template<typename T>
    class FALCOR_API SpectrumUI
    {
    public:
        SpectrumUI();
        SpectrumUI(const float2& wavelengthRange, const float2& spectralIntensityRange);
        void        setWavelengthRange(const float2& range) { mWavelengthRange = range; }
        void        setSpectralIntensityRange(const float2& range) { mSpectralIntensityRange = range; }
        bool        render(Gui::Widgets& w, const std::string name, std::vector<SampledSpectrum<T>*> spectra, const bool renderOnlySpectrum = false);
    protected:
        std::string makeUnique(const std::string& str) const;
        float       getSpectralIntensity(const float wavelength, const SampledSpectrum<T>* spectrum, const uint32_t curveIndex) const;
        float       getSpectralIntensity(const uint32_t pointIndex, const SampledSpectrum<T>* spectrum, const uint32_t curveIndex) const;
        uint32_t    getNumComponents() const;
        void        drawLine(ImDrawList* drawList, const float2& canvasPos, const float2& point0, const float2& point1, const float4& color, const float lineWidth = 2.0f);
        void        drawCircle(ImDrawList* drawList, const float2& canvasPos, const float2& center, const float radius, const float4& color);
        void        textHorizontallyCentered(const std::string& text, const float2& pos, const float4& color);
        void        textVerticallyCenteredLeft(const std::string& text, const float2& pos, const float4& color);
        float       toXCoord(const float wavelength, const float2& xAxisRange) const;
        float       toYCoord(const float spectralIntensity, const float2& yAxisRange) const;
        float2      toCoords(const float wavelength, const float spectralIntensity, const float2& xAxisRange, const float2& yAxisRange) const;
        float2      toCoords(const SampledSpectrum<T>* spectrum, const int index, const float2& xAxisRange, const float2& yAxisRange, const uint32_t float3Index = 0) const;
        void        drawColorMatchingFunctions(ImDrawList* drawList, const float2& canvasPos, const float2& canvasSize, const float2& xAxisRange, const float2& yAxisRange);
        void        drawTextWavelengthsAndTicks(ImDrawList* drawList, const float2& canvasPos, const float2& xAxisRange, const float2& yAxisRange, const float4& textColor, const float4& tickColor, const float4& gridColor);
        void        drawTextSpectralIntensityAndTicks(ImDrawList* drawList, const float2& canvasPos, const float2& xAxisRange, const float2& yAxisRange, const float4& textColor, const float4& tickColor, const float4& gridColor);
        void        drawSpectrumBar(ImDrawList* drawList, const float2& canvasPos, const float2& xAxisRange, const float2& yAxisRange, SampledSpectrum<T>* spectrum, const bool multiplyBySpectrum);
        void        drawSpectrumCurve(ImDrawList* drawList, const float2& canvasPos, const float2& canvasSize, const float2& xAxisRange, const float2& yAxisRange, SampledSpectrum<T>* spectrum, const uint32_t spectrumIndex);
        bool        handleMouse(const float2& canvasPos, const float2& canvasSize, const float2& xAxisRange, const float2& yAxisRange, ImDrawList* drawList, SampledSpectrum<T>* spectrum, const uint32_t float3Index = 0);
        float       generateNiceNumber(const float x) const;
    protected:
        // UI parameters.
        float2      mWavelengthRange = float2(350.0f, 750.0f);
        float2      mSpectralIntensityRange = float2(0.0f, 1.0f);
        uint32_t    mEditSpectrumIndex = 0;
        bool        mDrawSpectrumBar = true;
        bool        mMultiplyWithSpectralIntensity = true;
        bool        mDrawGridX = true;
        bool        mDrawGridY = true;
        uint32_t    mDrawAreaHeight = 300u;
        bool        mDrawColorMatchingFunctions = false;

        bool        mMovePoint = false;
        uint32_t    mPointIndexToBeEdited = 0u;
        SpectrumInterpolation mInterpolationType = SpectrumInterpolation::Linear;
    };

    template<typename T>
    bool renderSpectrumUI(Gui::Widgets& w, SampledSpectrum<T>& spectrum, const char label[] = "Spectrum UI")
    {
        SpectrumUI<T> spectrumUI;                                   // Use default parameters. Those will not be saved from frame to frame.
        return spectrumUI.render(w, label, { &spectrum }, true);    // True in the last parameter means that the UI for changing parameters will not be shown.
    }

    template<typename T>
    bool renderSpectrumUI(Gui::Widgets& w, SampledSpectrum<T>& spectrum, SpectrumUI<T>& spectrumUI, const char label[] = "Spectrum UI")
    {
        return spectrumUI.render(w, label, { &spectrum }, false);   // False in the last parameter means that the UI for changing parameters will be shown.
    }

    template<typename T>
    bool renderSpectrumUI(Gui::Widgets& w, std::vector<SampledSpectrum<T>*>& spectra, SpectrumUI<T>& spectrumUI, const char label[] = "Spectrum UI")
    {
        return spectrumUI.render(w, label, spectra, false);       // False in the last parameter means that the UI for changing parameters will be shown.
    }
}
