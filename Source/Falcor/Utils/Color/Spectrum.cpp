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
#include "Spectrum.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include <fstd/span.h> // TODO C++20: Replace with <span>
#include <unordered_map>

namespace Falcor
{
    // ------------------------------------------------------------------------
    // PiecewiseLinearSpectrum
    // ------------------------------------------------------------------------

    PiecewiseLinearSpectrum::PiecewiseLinearSpectrum(fstd::span<const float> wavelengths, fstd::span<const float> values)
        : mWavelengths(wavelengths.begin(), wavelengths.end())
        , mValues(values.begin(), values.end())
        , mMaxValue(*std::max_element(values.begin(), values.end()))
    {
        checkArgument(wavelengths.size() == values.size(), "'wavelengths' and 'values' need to contain the same number of elements");
    }

    PiecewiseLinearSpectrum PiecewiseLinearSpectrum::fromInterleaved(fstd::span<const float> interleaved, bool normalize)
    {
        checkArgument(interleaved.size() % 2 == 0, "'interleaved' must have an even number of elements.");

        size_t count = interleaved.size() / 2;
        std::vector<float> wavelengths(count);
        std::vector<float> values(count);

        for (size_t i = 0; i < count; ++i)
        {
            wavelengths[i] = interleaved[i * 2];
            values[i] = interleaved[i * 2 + 1];
            checkArgument(i == 0 || wavelengths[i] >= wavelengths[i - 1], "'interleaved' must have wavelengths that are monotonic increasing.");
        }

        auto spec = PiecewiseLinearSpectrum(wavelengths, values);

        if (normalize)
        {
            spec.scale(Spectra::kCIE_Y_Integral / innerProduct(spec, Spectra::kCIE_Y));
        }

        return spec;
    }

    std::optional<PiecewiseLinearSpectrum> PiecewiseLinearSpectrum::fromFile(const std::filesystem::path& path)
    {
        FALCOR_UNIMPLEMENTED();
    }

    void PiecewiseLinearSpectrum::scale(float factor)
    {
        checkArgument(factor >= 0.f, "'factor' ({}) needs to be positive.", factor);
        for (auto &value : mValues) value *= factor;
        mMaxValue *= factor;
    }

    // ------------------------------------------------------------------------
    // BlackbodySpectrum
    // ------------------------------------------------------------------------

    float blackbodyEmission(float wavelength, float temperature)
    {
        if (temperature <= 0.f) return 0.f;
        const float c = 299792458.f;
        const float h = 6.62606957e-34f;
        const float kb = 1.3806488e-23f;
        // Return emitted radiance for blackbody at wavelength lambda.
        float l = wavelength * 1e-9f;
        float Le = (2.f * h * c * c) / ((l * l * l * l * l) * (std::exp((h * c) / (l * kb * temperature)) - 1.f));
        FALCOR_ASSERT(!std::isnan(Le));
        return Le;
    }

    BlackbodySpectrum::BlackbodySpectrum(float temperature, bool normalize)
        : mTemperature(temperature)
    {
        // Compute wavelength (nm) of the peak of the spectrum using Wien's displacement law.
        float peakWavelength = (2.8977721e-3f / mTemperature) * 1e9f;

        // Setup normalization and max value constants.
        float peakValue = blackbodyEmission(peakWavelength, mTemperature);
        mNormalizationFactor = normalize ? 1.f / peakValue : 1.f;
        mMaxValue = normalize ? 1.f : peakValue;
    }

    namespace
    {
        #include "Spectra.inl"
    }

    const DenseleySampledSpectrum Spectra::kCIE_X(360.f, 830.f, CIE_X);
    const DenseleySampledSpectrum Spectra::kCIE_Y(360.f, 830.f, CIE_Y);
    const DenseleySampledSpectrum Spectra::kCIE_Z(360.f, 830.f, CIE_Z);

    namespace
    {
        const std::unordered_map<std::string, PiecewiseLinearSpectrum> kNamedSpectra
        {
            {
                "glass-BK7",
                PiecewiseLinearSpectrum::fromInterleaved(GlassBK7_eta, false)
            },
            {
                "glass-BAF10",
                PiecewiseLinearSpectrum::fromInterleaved(GlassBAF10_eta, false)
            },
            {
                "glass-FK51A",
                PiecewiseLinearSpectrum::fromInterleaved(GlassFK51A_eta, false)
            },
            {
                "glass-LASF9",
                PiecewiseLinearSpectrum::fromInterleaved(GlassLASF9_eta, false)
            },
            {
                "glass-F5",
                PiecewiseLinearSpectrum::fromInterleaved(GlassSF5_eta, false)
            },
            {
                "glass-F10",
                PiecewiseLinearSpectrum::fromInterleaved(GlassSF10_eta, false)
            },
            {
                "glass-F11",
                PiecewiseLinearSpectrum::fromInterleaved(GlassSF11_eta, false)
            },

            {
                "metal-Ag-eta",
                PiecewiseLinearSpectrum::fromInterleaved(Ag_eta, false)
            },
            {
                "metal-Ag-k",
                PiecewiseLinearSpectrum::fromInterleaved(Ag_k, false)
            },
            {
                "metal-Al-eta",
                PiecewiseLinearSpectrum::fromInterleaved(Al_eta, false)
            },
            {
                "metal-Al-k",
                PiecewiseLinearSpectrum::fromInterleaved(Al_k, false)
            },
            {
                "metal-Au-eta",
                PiecewiseLinearSpectrum::fromInterleaved(Au_eta, false)
            },
            {
                "metal-Au-k",
                PiecewiseLinearSpectrum::fromInterleaved(Au_k, false)
            },
            {
                "metal-Cu-eta",
                PiecewiseLinearSpectrum::fromInterleaved(Cu_eta, false)
            },
            {
                "metal-Cu-k",
                PiecewiseLinearSpectrum::fromInterleaved(Cu_k, false)
            },
            {
                "metal-CuZn-eta",
                PiecewiseLinearSpectrum::fromInterleaved(CuZn_eta, false)
            },
            {
                "metal-CuZn-k",
                PiecewiseLinearSpectrum::fromInterleaved(CuZn_k, false)
            },
            {
                "metal-MgO-eta",
                PiecewiseLinearSpectrum::fromInterleaved(MgO_eta, false)
            },
            {
                "metal-MgO-k",
                PiecewiseLinearSpectrum::fromInterleaved(MgO_k, false)
            },
            {
                "metal-TiO2-eta",
                PiecewiseLinearSpectrum::fromInterleaved(TiO2_eta, false)
            },
            {
                "metal-TiO2-k",
                PiecewiseLinearSpectrum::fromInterleaved(TiO2_k, false)
            },

            {
                "stdillum-A",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_A, true)
            },
            {
                "stdillum-D50",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_D5000, true)
            },
            {
                "stdillum-D65",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_D6500, true)
            },
            {
                "stdillum-F1",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F1, true)
            },
            {
                "stdillum-F2",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F2, true)
            },
            {
                "stdillum-F3",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F3, true)
            },
            {
                "stdillum-F4",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F4, true)
            },
            {
                "stdillum-F5",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F5, true)
            },
            {
                "stdillum-F6",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F6, true)
            },
            {
                "stdillum-F7",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F7, true)
            },
            {
                "stdillum-F8",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F8, true)
            },
            {
                "stdillum-F9",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F9, true)
            },
            {
                "stdillum-F10",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F10, true)
            },
            {
                "stdillum-F11",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F11, true)
            },
            {
                "stdillum-F12",
                PiecewiseLinearSpectrum::fromInterleaved(CIE_Illum_F12, true)
            },

            {
                "illum-acesD60",
                PiecewiseLinearSpectrum::fromInterleaved(ACES_Illum_D60, true)
            },

            {
                "canon_eos_100d_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_100d_r, false)
            },
            {
                "canon_eos_100d_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_100d_g, false)
            },
            {
                "canon_eos_100d_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_100d_b, false)
            },

            {
                "canon_eos_1dx_mkii_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_1dx_mkii_r, false)
            },
            {
                "canon_eos_1dx_mkii_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_1dx_mkii_g, false)
            },
            {
                "canon_eos_1dx_mkii_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_1dx_mkii_b, false)
            },

            {
                "canon_eos_200d_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_r, false)
            },
            {
                "canon_eos_200d_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_g, false)
            },
            {
                "canon_eos_200d_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_b, false)
            },

            {
                "canon_eos_200d_mkii_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_mkii_r, false)
            },
            {
                "canon_eos_200d_mkii_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_mkii_g, false)
            },
            {
                "canon_eos_200d_mkii_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_200d_mkii_b, false)
            },

            {
                "canon_eos_5d_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_r, false)
            },
            {
                "canon_eos_5d_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_g, false)
            },
            {
                "canon_eos_5d_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_b, false)
            },

            {
                "canon_eos_5d_mkii_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkii_r, false)
            },
            {
                "canon_eos_5d_mkii_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkii_g, false)
            },
            {
                "canon_eos_5d_mkii_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkii_b, false)
            },

            {
                "canon_eos_5d_mkiii_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiii_r, false)
            },
            {
                "canon_eos_5d_mkiii_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiii_g, false)
            },
            {
                "canon_eos_5d_mkiii_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiii_b, false)
            },

            {
                "canon_eos_5d_mkiv_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiv_r, false)
            },
            {
                "canon_eos_5d_mkiv_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiv_g, false)
            },
            {
                "canon_eos_5d_mkiv_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5d_mkiv_b, false)
            },

            {
                "canon_eos_5ds_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5ds_r, false)
            },
            {
                "canon_eos_5ds_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5ds_g, false)
            },
            {
                "canon_eos_5ds_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_5ds_b, false)
            },

            {
                "canon_eos_m_r",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_m_r, false)
            },
            {
                "canon_eos_m_g",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_m_g, false)
            },
            {
                "canon_eos_m_b",
                PiecewiseLinearSpectrum::fromInterleaved(canon_eos_m_b, false)
            },

            {
                "hasselblad_l1d_20c_r",
                PiecewiseLinearSpectrum::fromInterleaved(hasselblad_l1d_20c_r, false)
            },
            {
                "hasselblad_l1d_20c_g",
                PiecewiseLinearSpectrum::fromInterleaved(hasselblad_l1d_20c_g, false)
            },
            {
                "hasselblad_l1d_20c_b",
                PiecewiseLinearSpectrum::fromInterleaved(hasselblad_l1d_20c_b, false)
            },

            {
                "nikon_d810_r",
                PiecewiseLinearSpectrum::fromInterleaved(nikon_d810_r, false)
            },
            {
                "nikon_d810_g",
                PiecewiseLinearSpectrum::fromInterleaved(nikon_d810_g, false)
            },
            {
                "nikon_d810_b",
                PiecewiseLinearSpectrum::fromInterleaved(nikon_d810_b, false)
            },

            {
                "nikon_d850_r",
                PiecewiseLinearSpectrum::fromInterleaved(nikon_d850_r, false)
            },
            {
                "nikon_d850_g",
                PiecewiseLinearSpectrum::fromInterleaved(nikon_d850_g, false)
            },
            {
                "nikon_d850_b",
                PiecewiseLinearSpectrum::fromInterleaved(nikon_d850_b, false)
            },

            {
                "sony_ilce_6400_r",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_6400_r, false)
            },
            {
                "sony_ilce_6400_g",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_6400_g, false)
            },
            {
                "sony_ilce_6400_b",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_6400_b, false)
            },

            {
                "sony_ilce_7m3_r",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7m3_r, false)
            },
            {
                "sony_ilce_7m3_g",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7m3_g, false)
            },
            {
                "sony_ilce_7m3_b",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7m3_b, false)
            },

            {
                "sony_ilce_7rm3_r",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7rm3_r, false)
            },
            {
                "sony_ilce_7rm3_g",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7rm3_g, false)
            },
            {
                "sony_ilce_7rm3_b",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_7rm3_b, false)
            },

            {
                "sony_ilce_9_r",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_9_r, false)
            },
            {
                "sony_ilce_9_g",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_9_g, false)
            },
            {
                "sony_ilce_9_b",
                PiecewiseLinearSpectrum::fromInterleaved(sony_ilce_9_b, false)
            }
        };
    }

    const PiecewiseLinearSpectrum* Spectra::getNamedSpectrum(const std::string& name)
    {
        auto it = kNamedSpectra.find(name);
        if (it == kNamedSpectra.end()) return nullptr;
        return &it->second;
    }
}
