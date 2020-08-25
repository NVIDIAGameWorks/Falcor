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
#pragma once
#include "Core/Program/Program.h"
#include "SampleGeneratorType.slangh"

namespace Falcor
{
    /** Utility class for sample generators on the GPU.

        This class has functions for configuring the shader program and
        uploading the necessary lookup tables (if needed).
        On the GPU, import SampleGenerator.slang in your shader program.
    */
    class dlldecl SampleGenerator : public std::enable_shared_from_this<SampleGenerator>
    {
    public:
        using SharedPtr = std::shared_ptr<SampleGenerator>;
        using SharedConstPtr = std::shared_ptr<const SampleGenerator>;

        virtual ~SampleGenerator() = default;

        /** Factory function for creating a sample generator of the specified type.
            \param[in] type The type of sample generator. See SampleGeneratorType.slangh.
            \return New object, or throws an exception on error.
        */
        static SharedPtr create(uint32_t type);

        /** Get macro definitions for this sample generator.
            \return Macro definitions that must be set on the shader program that uses this sampler.
        */
        virtual Shader::DefineList getDefines() const;

        /** Binds the data to a program vars object.
            \param[in] pVars ProgramVars of the program to set data into.
            \return false if there was an error, true otherwise.
        */
        virtual bool setShaderData(ShaderVar const& var) const { return true; }

        /** Returns a GUI dropdown list of all available sample generators.
        */
        static const Gui::DropdownList& getGuiDropdownList();

        /** Register a sample generator type.
            \param[in] type The type of sample generator. See SampleGeneratorType.slangh.
            \param[in] name Descriptive name used in the UI.
            \param[in] createFunc Function to create an instance of the sample generator.
        */
        static void registerType(uint32_t type, const std::string& name, std::function<SharedPtr()> createFunc);

    protected:
        SampleGenerator(uint32_t type) : mType(type) {}

        const uint32_t mType;       ///< Type of sample generator. See SampleGeneratorType.slangh.

    private:
        /** Register all basic sample generator types.
        */
        static void registerAll();

        friend struct RegisterSampleGenerators;
    };
}
