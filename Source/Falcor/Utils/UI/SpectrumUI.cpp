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
#include "stdafx.h"
#include "Utils/UI/SpectrumUI.h"

namespace Falcor
{
    template<typename T>
    SpectrumUI<T>::SpectrumUI()
    {
    }

    template<typename T>
    SpectrumUI<T>::SpectrumUI(const float2& wavelengthRange, const float2& spectralIntensityRange)
    {
        mWavelengthRange = wavelengthRange;
        mSpectralIntensityRange = spectralIntensityRange;
    }

    // Function for making a ImGui string unique by adding an invisible (##) number (the address of this) to the string.
    template<typename T>
    std::string SpectrumUI<T>::makeUnique(const std::string& str) const
    {
        return str + "##" + std::to_string(reinterpret_cast<std::uintptr_t>(this));
    }

    // Get spectral intensity at a certain wavelength for a spectrum, with curveIndex selecting one of the three components if T=float3.
    template<typename T>
    float SpectrumUI<T>::getSpectralIntensity(const float wavelength, const SampledSpectrum<T>* spectrum, const uint32_t curveIndex) const
    {
        T spectralIntensity = spectrum->eval(wavelength, mInterpolationType);
        float s = 0.0f;
        if constexpr (std::is_same_v<T, float>)
        {
            s = spectralIntensity;
        }
        else if constexpr (std::is_same_v<T, float3>)
        {
            s = spectralIntensity[curveIndex % 3];
        }
        else
        {
            FALCOR_ASSERT(false);
        }
        return s;
    }

    // Get spectral intensity at a certain index for a spectrum, with curveIndex selecting one of the three components if T=float3.
    template<typename T>
    float SpectrumUI<T>::getSpectralIntensity(const uint32_t pointIndex, const SampledSpectrum<T>* spectrum, const uint32_t curveIndex) const
    {
        T spectralIntensity = spectrum->get(pointIndex);
        float s = 0.0f;
        if constexpr (std::is_same_v<T, float>)
        {
            s = spectralIntensity;
        }
        else if constexpr (std::is_same_v<T, float3>)
        {
            s = spectralIntensity[curveIndex % 3];
        }
        else
        {
            FALCOR_ASSERT(false);
        }
        return s;
    }

    // Get the number of components of T.
    template<typename T>
    uint32_t SpectrumUI<T>::getNumComponents() const
    {
        if constexpr (std::is_same_v<T, float3>)
        {
            return 3u;
        }
        return 1u;
    }

    // Utility function for rendering text centered horizontally.
    template<typename T>
    void SpectrumUI<T>::textHorizontallyCentered(const std::string& text, const float2& pos, const float4& color)
    {
        float strWidth = ImGui::CalcTextSize(text.c_str()).x;
        ImGui::SetCursorPosX(pos.x - strWidth * 0.5f);
        ImGui::SetCursorPosY(pos.y);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(color.x, color.y, color.z, color.w));
        ImGui::Text(text.c_str());
        ImGui::PopStyleColor();
    }

    // Utility function for rendering text centered vertically to the left of a certain position.
    template<typename T>
    void SpectrumUI<T>::textVerticallyCenteredLeft(const std::string& text, const float2& pos, const float4& color)
    {
        ImVec2 strSize = ImGui::CalcTextSize(text.c_str());
        ImGui::SetCursorPosX(pos.x - strSize.x);
        ImGui::SetCursorPosY(pos.y - strSize.y * 0.5f);
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(color.x, color.y, color.z, color.w));
        ImGui::Text(text.c_str());
        ImGui::PopStyleColor();
    }

    // Utility function for drawing a line with ImGui.
    template<typename T>
    void SpectrumUI<T>::drawLine(ImDrawList* drawList, const float2& canvasPos, const float2& point0, const float2& point1, const float4& color, const float lineWidth)
    {
        drawList->AddLine(ImVec2(canvasPos.x + point0.x, canvasPos.y + point0.y), ImVec2(canvasPos.x + point1.x, canvasPos.y + point1.y), ImColor(color.x, color.y, color.z, color.w), lineWidth);
    }

    // Utility function for drawing a filled circle with ImGui.
    template<typename T>
    void SpectrumUI<T>::drawCircle(ImDrawList* drawList, const float2& canvasPos, const float2& center, const float radius, const float4& color)
    {
        drawList->AddCircleFilled(ImVec2(canvasPos.x + center.x, canvasPos.y + center.y), radius, ImColor(color.x, color.y, color.z, color.w));
    }

    // Convert from wavelengh to coordinate inside ImGui.
    template<typename T>
    float SpectrumUI<T>::toXCoord(const float wavelength, const float2& xAxisRange) const
    {
        const float t = (wavelength - mWavelengthRange.x) / (mWavelengthRange.y - mWavelengthRange.x);    // In [0,1].
        return  xAxisRange.x + t * (xAxisRange.y - xAxisRange.x);
    }

    // Convert from spectral intensity to coordinate inside ImGui.
    template<typename T>
    float SpectrumUI<T>::toYCoord(const float spectralIntensity, const float2& yAxisRange) const
    {
        const float t = (spectralIntensity - mSpectralIntensityRange.x) / (mSpectralIntensityRange.y - mSpectralIntensityRange.x);    // In [0,1].
        return  yAxisRange.x + t * (yAxisRange.y - yAxisRange.x);
    }

    // Convert from (wavelengh, spectral intensity) to coordinates inside ImGui.
    template<typename T>
    float2 SpectrumUI<T>::toCoords(const float wavelength, const float spectralIntensity, const float2& xAxisRange, const float2& yAxisRange) const
    {
        return float2(toXCoord(wavelength, xAxisRange), toYCoord(spectralIntensity, yAxisRange));
    }

    // Convert a certain point (with index) from the Spectrum to coordinates inside ImGui.
    template<typename T>
    float2 SpectrumUI<T>::toCoords(const SampledSpectrum<T>* spectrum, const int index, const float2& xAxisRange, const float2& yAxisRange, const uint32_t float3Index) const
    {
        const int idx = std::clamp(index, 0, int(spectrum->size() - 1));
        const float deltaW = (spectrum->getWavelengthRange().y - spectrum->getWavelengthRange().x) / (spectrum->size() - 1.0f);
        const float wavelength = spectrum->getWavelengthRange().x + idx * deltaW;

        T spectralIntensity = spectrum->get(idx);
        if constexpr (std::is_same_v<T, float>)
        {
            return toCoords(wavelength, spectralIntensity, xAxisRange, yAxisRange);
        }
        else if constexpr (std::is_same_v<T, float3>)
        {
            return toCoords(wavelength, spectralIntensity[float3Index], xAxisRange, yAxisRange);
        }
        FALCOR_ASSERT(false);   // Should never happen.
    }

    // Drawing all three color matching functions.
    template<typename T>
    void SpectrumUI<T>::drawColorMatchingFunctions(ImDrawList* drawList, const float2& canvasPos, const float2& canvasSize, const float2& xAxisRange, const float2& yAxisRange)
    {
        if (!mDrawColorMatchingFunctions)
        {
            return;
        }
        const float4 R = float4(0.8f, 0.0f, 0.0f, 1.0f);
        const float4 G = float4(0.0f, 0.7f, 0.0f, 1.0f);
        const float4 B = float4(0.0f, 0.0f, 1.0f, 1.0f);
        const uint32_t loop = uint32_t(canvasSize.x / 4.0f);
        const float waveLengthDelta = (mWavelengthRange.y - mWavelengthRange.x) / (loop - 1.0f);

        float4 P, pPrev;
        for (uint32_t q = 0; q <= loop; q++)
        {
            float wavelength = mWavelengthRange.x + waveLengthDelta * q;
            float3 xyzBar = SpectrumUtils::wavelengthToXYZ_CIE1931(wavelength);
            float2 tmp = toCoords(wavelength, xyzBar.x, xAxisRange, yAxisRange);
            P.w = tmp.x;
            P.x = tmp.y;
            P.y = toCoords(wavelength, xyzBar.y, xAxisRange, yAxisRange).y;
            P.z = toCoords(wavelength, xyzBar.z, xAxisRange, yAxisRange).y;
            if (q > 0)
            {
                drawLine(drawList, canvasPos, float2(pPrev.w, pPrev.x), float2(P.w, P.x), R, 1.0f);
                drawLine(drawList, canvasPos, float2(pPrev.w, pPrev.y), float2(P.w, P.y), G, 1.0f);
                drawLine(drawList, canvasPos, float2(pPrev.w, pPrev.z), float2(P.w, P.z), B, 1.0f);
            }
            pPrev = P;
        }
    }

    // Draw ticks on the x-axis and text below.
    template<typename T>
    void SpectrumUI<T>::drawTextWavelengthsAndTicks(ImDrawList* drawList, const float2& canvasPos, const float2& xAxisRange, const float2& yAxisRange, const float4& textColor, const float4& tickColor, const float4& gridColor)
    {
        const float halfTickSize = 4.0f;
        auto spreadWavelengthTexts = [&](ImDrawList* drawList, const uint32_t numTicks, const float2& canvasPos, const float2& xAxisRange, const float2& yAxisRange)
        {
            for (uint32_t q = 0; q < numTicks; q++)
            {
                float wavelength = q * (mWavelengthRange.y - mWavelengthRange.x) / (numTicks - 1.0f) + mWavelengthRange.x;
                float x = toXCoord(wavelength, xAxisRange);
                textHorizontallyCentered(std::to_string(uint32_t(wavelength)), float2(x, yAxisRange.x + 2.0f), textColor);
                if (mDrawGridX)
                {
                    drawLine(drawList, canvasPos, float2(x, yAxisRange.x), float2(x, yAxisRange.y), gridColor, 1.0f);
                }
                drawLine(drawList, canvasPos, float2(x, yAxisRange.x - halfTickSize), float2(x, yAxisRange.x + halfTickSize), tickColor, 2.0f);
            }
        };

        float strWidth = ImGui::CalcTextSize("000").x;
        float w = xAxisRange.y - xAxisRange.x;
        float diff = mWavelengthRange.y - mWavelengthRange.x;

        // The second argument to all spreadFrequencyTexts() below have been tailored especially for the case when the wavelengthXAxisRange = (350,750) and (400,700). Can be generalized.
        if (w <= 4.0f * strWidth)
        {
            spreadWavelengthTexts(drawList, 2, canvasPos, xAxisRange, yAxisRange);
        }
        else if (w <= 8.0f * strWidth)
        {
            spreadWavelengthTexts(drawList, 3, canvasPos, xAxisRange, yAxisRange);
        }
        else if (w <= 16.0f * strWidth)
        {
            spreadWavelengthTexts(drawList, diff <= 300.0f ? 4 : 5, canvasPos, xAxisRange, yAxisRange);
        }
        else if (w <= 32.0f * strWidth)
        {
            spreadWavelengthTexts(drawList, diff <= 300.0f ? 7 : (diff <= 350.0f ? 8 : 9), canvasPos, xAxisRange, yAxisRange);
        }
        else
        {
            spreadWavelengthTexts(drawList, diff <= 300.0f ? 13 : (diff <= 350.0f ? 15 : 17), canvasPos, xAxisRange, yAxisRange);
        }
    }

    // Draw ticks on the y-axis and text to the left.
    template<typename T>
    void SpectrumUI<T>::drawTextSpectralIntensityAndTicks(ImDrawList* drawList, const float2& canvasPos, const float2& xAxisRange, const float2& yAxisRange, const float4& textColor, const float4& tickColor, const float4& gridColor)
    {
        const float halfTickSize = 4.0f;
        const float strHeight = ImGui::CalcTextSize("0").y;
        const float h = yAxisRange.x - yAxisRange.y;                                // In pixels (x >= y) since y points downwards.
        const float sw = mSpectralIntensityRange.y - mSpectralIntensityRange.x;     // In pixels.
        const uint32_t maxNumTicks = uint32_t(h / strHeight) / 2;                           // With spacing of one strHeight inbetween.
        const uint32_t maxTicksIn0to1 = uint32_t(h / (sw * strHeight)) / 2;

        uint32_t numTicks;
        float delta;
        if (maxTicksIn0to1 == 0)
        {
            numTicks = std::max(1u, maxNumTicks);
            delta = float(uint32_t(sw)) / numTicks;
        }
        else if (maxTicksIn0to1 == 1)
        {
            numTicks = uint32_t(sw);
            delta = 1.0f;
        }
        else if (maxTicksIn0to1 <= 9)
        {
            numTicks = uint32_t(sw * 2.0f);
            delta = 0.5f;
        }
        else
        {
            numTicks = uint32_t(sw * 10.0f);
            delta = 0.1f;
        }

        char str[20];
        for (uint32_t q = 0; q < numTicks + 1; q++)
        {
            float spectralIntensity = q * delta + mSpectralIntensityRange.x;
            float y = toYCoord(spectralIntensity, yAxisRange);
            sprintf_s(str, "%1.1f", spectralIntensity);
            textVerticallyCenteredLeft(q == 0.0 ? "0" : std::string(str), float2(xAxisRange.x - (q == 0 ? 15.0f : 7.0f), y), textColor);
            if (mDrawGridY)
            {
                drawLine(drawList, canvasPos, float2(xAxisRange.x, y), float2(xAxisRange.y, y), gridColor, 1.0f);
            }
            drawLine(drawList, canvasPos, float2(xAxisRange.x - halfTickSize, y), float2(xAxisRange.x + halfTickSize, y), tickColor, 2.0f);
        }
    }

    // Draw a spectrum bar where the x-axis is.
    template<typename T>
    void SpectrumUI<T>::drawSpectrumBar(ImDrawList* drawList, const float2& canvasPos, const float2& xAxisRange, const float2& yAxisRange, SampledSpectrum<T>* spectrum, const bool multiplyBySpectrum)
    {
        const float w = mWavelengthRange.y - mWavelengthRange.x;
        uint32_t xs = uint32_t(xAxisRange.x + (xAxisRange.y - xAxisRange.x) * (std::max(spectrum->getWavelengthRange().x, mWavelengthRange.x) - mWavelengthRange.x) / w);
        uint32_t xe = uint32_t(xAxisRange.x + (xAxisRange.y - xAxisRange.x) * (std::min(spectrum->getWavelengthRange().y, mWavelengthRange.y) - mWavelengthRange.x) / w);
        for (uint32_t x = xs; x <= xe; x++)
        {
            float wavelength = w * (x - xAxisRange.x) / (xAxisRange.y - xAxisRange.x) + mWavelengthRange.x;
            float spectralIntensity = getSpectralIntensity(wavelength, spectrum, mEditSpectrumIndex);
            float3 color = SpectrumUtils::wavelengthToRGB_Rec709(wavelength) * (multiplyBySpectrum ? spectralIntensity : 1.0f);
            drawLine(drawList, canvasPos, float2(x, yAxisRange.x - 3.0f), float2(x, yAxisRange.x + 3.0f), float4(sRGBToLinear(color), 1.0f), 1.0f);
        }
    }

    // Draw the spectrum curve with lines and circles at the samples with the chosen type of interpolation.
    template<typename T>
    void SpectrumUI<T>::drawSpectrumCurve(ImDrawList* drawList, const float2& canvasPos, const float2& canvasSize, const float2& xAxisRange, const float2& yAxisRange,
        SampledSpectrum<T>* spectrum, const uint32_t spectrumIndex)
    {
        const float deltaW = (spectrum->getWavelengthRange().y - spectrum->getWavelengthRange().x) / (spectrum->size() - 1.0f);

        // This loop is 1 for T=float and 3 when T=float3.
        uint32_t numComponents = getNumComponents();
        float4 lineColor;
        for (uint32_t index = 0; index < numComponents; index++)
        {
            bool lineNeedHighlight = spectrumIndex * numComponents + index == mEditSpectrumIndex;
            float lineWidth = lineNeedHighlight ? 3.0f : 2.0f;
            if (numComponents == 3)
            {
                const float4 colors[] = { float4(0.8, 0.0f, 0.0f, 1.0f), float4(0.0, 0.7f, 0.0f, 1.0f), float4(0.0, 0.0f, 1.0f, 1.0f) };
                lineColor = colors[index] * (lineNeedHighlight ? 1.0f : 0.7f);
            }
            else
            {
                lineColor = float4(0.8f, 0.8f, 0.8f, 1.0f) * (lineNeedHighlight ? 1.0f : 0.5f);
            }
            // Draw the curve between two points.
            for (uint32_t q = 1; q < spectrum->size(); q++)
            {
                float2 p0 = toCoords(spectrum, q - 1, xAxisRange, yAxisRange, index);
                float2 p1 = toCoords(spectrum, q, xAxisRange, yAxisRange, index);
                const uint32_t numLines = 10;
                if (mInterpolationType == SpectrumInterpolation::Linear)
                {
                    drawLine(drawList, canvasPos, p0, p1, lineColor, lineWidth);
                }
                else
                {
                    FALCOR_ASSERT(false);   // WIP: Will add more modeas in another MR.
                }
            }
        }
        for (uint32_t index = 0; index < getNumComponents(); index++)
        {
            // Draw the points.
            const float radius = 5.0f;
            for (uint32_t q = 0; q < spectrum->size(); q++)
            {
                float wavelength = spectrum->getWavelengthRange().x + q * deltaW;
                float spectralIntensity = (q == uint32_t(spectrum->size() - 1)) ? getSpectralIntensity(q, spectrum, index) : getSpectralIntensity(wavelength, spectrum, index);
                float2 P = toCoords(wavelength, spectralIntensity, xAxisRange, yAxisRange);
                float3 color = SpectrumUtils::wavelengthToRGB_Rec709(wavelength) * (mMultiplyWithSpectralIntensity ? spectralIntensity : 1.0f);
                drawCircle(drawList, canvasPos, P, radius, float4(sRGBToLinear(color), 1.0f));
            }
        }
    }

    // Handle all the mouse interactions: moving the spectrumm points.
    template<typename T>
    bool SpectrumUI<T>::handleMouse(const float2& canvasPos, const float2& canvasSize, const float2& xAxisRange, const float2& yAxisRange, ImDrawList* drawList, SampledSpectrum<T>* spectrum, const uint32_t float3Index)
    {
        bool changed = false;
        if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_AllowWhenBlockedByActiveItem))
        {
            return false;
        }
        float2 mousePosCanvas = float2(ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y) - canvasPos;

        const float nearRadius = 30.0f;
        if (!mMovePoint)                                    // No point has been clicked.
        {
            float smallestDist2 = std::numeric_limits<float>::max();
            mPointIndexToBeEdited = 0u;
            for (uint32_t q = 0; q < spectrum->size(); q++)      // Without caring about whether the user has clicked, search for the point closest to the mouse (within nearRadius).
            {
                float2 p = toCoords(spectrum, q, xAxisRange, yAxisRange, float3Index);
                float2 diff = mousePosCanvas - p;
                float dist2 = dot(diff, diff);
                if (dist2 < nearRadius * nearRadius && dist2 < smallestDist2)
                {
                    mMovePoint = true;
                    smallestDist2 = dist2;
                    mPointIndexToBeEdited = q;
                }
            }
            if (mMovePoint)                                 // If the user hovered or clicked on a point sufficiently close, then draw a circle around the point.
            {
                float2 p = toCoords(spectrum, mPointIndexToBeEdited, xAxisRange, yAxisRange, float3Index);
                drawCircle(drawList, canvasPos, p, nearRadius, float4(1.0f, 1.0f, 1.0f, 0.1f));
                char str[200];
                sprintf_s(str, "%2.4f", getSpectralIntensity(mPointIndexToBeEdited, spectrum, float3Index));
                ImGui::SetCursorPosX(p.x + 20.0f);
                ImGui::SetCursorPosY(p.y - ImGui::CalcTextSize("0").y * 0.5f);
                ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32_WHITE);
                ImGui::Text(str);
                ImGui::PopStyleColor();

                mMovePoint = ImGui::GetIO().MouseDown[0];       // Only let the user move the point, if the left mouse is down.
            }
        }
        if (mMovePoint)                                         // Move the clicked point.
        {

            float prevSpectralIntensity = getSpectralIntensity(mPointIndexToBeEdited, spectrum, float3Index);
            float spectralIntensity = (yAxisRange.x - mousePosCanvas.y) / (yAxisRange.x - yAxisRange.y) * (mSpectralIntensityRange.y - mSpectralIntensityRange.x) + mSpectralIntensityRange.x;
            spectralIntensity = std::max(0.0f, spectralIntensity);

            if constexpr (std::is_same_v<T, float3>)
            {
                float3 p = spectrum->get(mPointIndexToBeEdited);
                p[float3Index] = spectralIntensity;
                spectrum->set(mPointIndexToBeEdited, p);
            }
            else
            {
                spectrum->set(mPointIndexToBeEdited, spectralIntensity);
            }
            changed = prevSpectralIntensity != spectralIntensity;

            float2 p = toCoords(spectrum, mPointIndexToBeEdited, xAxisRange, yAxisRange);
            drawCircle(drawList, canvasPos, p, nearRadius, float4(1.0f, 1.0f, 1.0f, 0.1f)); // Draw a circle around the point being moved.

            char str[200];
            sprintf_s(str, "%2.4f", getSpectralIntensity(mPointIndexToBeEdited, spectrum, float3Index));
            ImGui::SetCursorPosX(p.x + 20.0f);
            ImGui::SetCursorPosY(p.y - ImGui::CalcTextSize("0").y * 0.5f);
            ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32_WHITE);
            ImGui::Text(str);
            ImGui::PopStyleColor();

            mMovePoint = ImGui::GetIO().MouseDown[0];       // Only let the user continue to move the point, if the left mouse is down.
        }
        return changed;
    }

    // Rendering the spectrum. Note that this function can render several Spectrum on the same drawing area, which can be convenient for debugging purposes & algorithm development.
    template<typename T>
    bool SpectrumUI<T>::render(Gui::Widgets& w, const std::string name, std::vector<SampledSpectrum<T>*> spectra, const bool renderOnlySpectrum)
    {
        if (spectra.size() == 0)
        {
            return false;
        }
        const uint32_t numComponents = getNumComponents();
        Gui::Group mainGroup = w.group(name, true);
        if (!mainGroup.open())
        {
            return false;
        }

        bool changed = false;
        if (!renderOnlySpectrum)                // Note: cannot combine this if-case with the next, since we do not want group() to be called if renderOnlySpectrum == true.
        {
            Gui::Group guiGroup = mainGroup.group(makeUnique("UI").c_str(), true);
            if (guiGroup.open())
            {
                uint32_t idx = uint32_t(mInterpolationType);
                guiGroup.dropdown(makeUnique("Interpolation").c_str(), kInterpolationDropdownList, idx);
                changed |= (idx != uint32_t(mInterpolationType));
                mInterpolationType = SpectrumInterpolation(idx);

                changed |= guiGroup.checkbox(makeUnique("Draw spectrum bar").c_str(), mDrawSpectrumBar, false);
                if (mDrawSpectrumBar)
                {
                    changed |= guiGroup.checkbox(makeUnique("Multiply with spectral intensity").c_str(), mMultiplyWithSpectralIntensity, true);
                }
                changed |= guiGroup.var(makeUnique("Height").c_str(), mDrawAreaHeight, 64u, 2048u, 1u, false);
                if (spectra.size() * numComponents > 1)
                {
                    changed |= guiGroup.var(makeUnique("Index to editable curve").c_str(), mEditSpectrumIndex, 0u, uint32_t(spectra.size() * numComponents - 1), 1u, true);
                }
                changed |= guiGroup.checkbox(makeUnique("Vertical grid").c_str(), mDrawGridX, false);
                changed |= guiGroup.checkbox(makeUnique("Horizontal grid").c_str(), mDrawGridY, true);
                changed |= guiGroup.checkbox(makeUnique("Color matching functions").c_str(), mDrawColorMatchingFunctions, true);
                changed |= guiGroup.var(makeUnique("Min wavelength").c_str(), mWavelengthRange.x, 0.0f, 800.0f, 5.0f);
                changed |= guiGroup.var(makeUnique("Max wavelength").c_str(), mWavelengthRange.y, 0.0f, 800.0f, 5.0f, true);
                if (mWavelengthRange.x > mWavelengthRange.y)
                {
                    mWavelengthRange.y = mWavelengthRange.x + 5.0f;
                }
                changed |= guiGroup.var(makeUnique("Max spectral intensity").c_str(), mSpectralIntensityRange.y, 1.0f, 10.0f, 0.5f);

                guiGroup.release();
            }
        }

        // Draw colored boxes of the spectrum --> RGB for each in spectra and the RGB color in text below.
        const float rgbWidth = ImGui::CalcTextSize("0.00, 0.00, 0.00").x;
        const float rgbHeight = 20.0f;
        const float separation = 5.0f;
        char str[200] = "";
        ImVec2 p = ImGui::GetCursorScreenPos();
        ImGui::Dummy(ImVec2(0, 20 + ImGui::CalcTextSize("0").y + separation));
        for (uint32_t q = 0; q < uint32_t(spectra.size()); q++)
        {
            for (uint32_t index = 0; index < numComponents; index++)
            {
                float3 c = SpectrumUtils::toRGB_D65<T>(*spectra[q], mInterpolationType, index);
                sprintf_s(str, "%1.2f, %1.2f, %1.2f", c.x, c.y, c.z);
                c = sRGBToLinear(c);
                ImGui::GetWindowDrawList()->AddRectFilled(p, ImVec2(p.x + rgbWidth, p.y + rgbHeight), ImColor(c.r, c.g, c.b, 1.0f));
                ImGui::GetWindowDrawList()->AddText(ImVec2(p.x, p.y + rgbHeight + separation), IM_COL32_WHITE, str);
                p.x += separation + rgbWidth;
            }
        }

        const float marginXLeft = 50.0f;
        const float marginXRight = 20.0f;
        const float marginYTop = 10.0f;
        const float marginYBottom = 30.0f;
        const float4 lightGray = float4(0.75f, 0.75f, 0.75f, 1.0f);
        const float4 gray = float4(0.5f, 0.5f, 0.5f, 1.0f);

        // This is the main drawing area of the spectrum visualization.
        ImGui::BeginChild(makeUnique("Spectrum visualization").c_str(), ImVec2(ImGui::GetWindowContentRegionWidth(), float(mDrawAreaHeight)), false, ImGuiWindowFlags_NoScrollWithMouse);
        {
            const float2 canvasPos = float2(ImGui::GetCursorScreenPos().x, ImGui::GetCursorScreenPos().y);            // ImDrawList API uses screen coordinates.
            const float2 canvasSize = float2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y);
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            mEditSpectrumIndex = std::clamp(mEditSpectrumIndex, 0u, uint32_t(spectra.size() * numComponents - 1));

            ImGui::InvisibleButton(makeUnique("canvas").c_str(), ImVec2(canvasSize.x, canvasSize.y));   // Added so that the ImGUI window cannot be moved by clicking on the SpectrumUI. We want to consume these click by our selves.

            // Draw border. ImGui can do this with third parameter to BeginChild(), but it clears out a thing band where you cannot draw, so better to do it this way.
            drawLine(drawList, canvasPos, float2(0.0, 0.0f), float2(canvasSize.x, 0.0f), lightGray, 1.0f);
            drawLine(drawList, canvasPos, float2(0.0, canvasSize.y - 1.0f), float2(canvasSize.x, canvasSize.y - 1.0f), lightGray, 1.0f);
            drawLine(drawList, canvasPos, float2(0.0, 0.0f), float2(0.0, canvasSize.y - 1.0f), lightGray, 1.0f);
            drawLine(drawList, canvasPos, float2(canvasSize.x - 1.0f, 0.0f), float2(canvasSize.x - 1.0f, canvasSize.y - 1.0f), lightGray, 1.0f);
            const float2 xAxisRange = float2(marginXLeft, canvasSize.x - marginXRight);
            const float2 yAxisRange = float2(canvasSize.y - marginYBottom, marginYTop);
            if (mDrawSpectrumBar)
            {
                drawSpectrumBar(drawList, canvasPos, xAxisRange, yAxisRange, spectra[mEditSpectrumIndex / numComponents], mMultiplyWithSpectralIntensity);
            }

            drawLine(drawList, canvasPos, float2(xAxisRange.x, yAxisRange.x), float2(xAxisRange.y, yAxisRange.x), gray, 1.0f);  // Draw x-axis.
            drawLine(drawList, canvasPos, float2(xAxisRange.x, yAxisRange.x), float2(xAxisRange.x, yAxisRange.y), gray, 1.0f);  // Draw y-axis.
            drawLine(drawList, canvasPos, float2(xAxisRange.y, yAxisRange.x), float2(xAxisRange.y, yAxisRange.y), gray, 1.0f);  // Draw y-axis to the right.

            drawColorMatchingFunctions(drawList, canvasPos, canvasSize, xAxisRange, yAxisRange);                                // Draw the color matching functions xyzbar.
            drawTextWavelengthsAndTicks(drawList, canvasPos, xAxisRange, yAxisRange, gray, lightGray, lightGray * 0.5f);        // Draw the text wavelengths below the x-axis and the ticks.
            drawTextSpectralIntensityAndTicks(drawList, canvasPos, xAxisRange, yAxisRange, gray, lightGray, lightGray * 0.5f);  // Draw the ticks on the y-axis and spectral intensity text there.

            // Draw spectrum curves.
            for (uint32_t q = 0; q < uint32_t(spectra.size()); q++)
            {
                drawSpectrumCurve(drawList, canvasPos, canvasSize, xAxisRange, yAxisRange, spectra[q], q);
            }
            uint32_t float3Index = numComponents == 1 ? 0 : (mEditSpectrumIndex % 3);
            changed |= handleMouse(canvasPos, canvasSize, xAxisRange, yAxisRange, drawList, spectra[mEditSpectrumIndex / numComponents], float3Index);
        }
        ImGui::EndChild();
        mainGroup.release();
        return changed;
    }

    template class SpectrumUI<float>;
    template class SpectrumUI<float3>;
}
