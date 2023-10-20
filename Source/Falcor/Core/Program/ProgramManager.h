/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Program.h"
#include "Core/Macros.h"
#include "Core/API/fwd.h"

#include <memory>

namespace Falcor
{

class FALCOR_API ProgramManager
{
public:
    ProgramManager(Device* pDevice);

    /**
     * Defines flags that should be forcefully disabled or enabled on all shaders.
     * When a flag is in both groups, it gets enabled.
     */
    struct ForcedCompilerFlags
    {
        SlangCompilerFlags enabled = SlangCompilerFlags::None;  ///< Compiler flags forcefully enabled on all shaders
        SlangCompilerFlags disabled = SlangCompilerFlags::None; ///< Compiler flags forcefully enabled on all shaders
    };

    struct CompilationStats
    {
        size_t programVersionCount = 0;
        size_t programKernelsCount = 0;
        double programVersionMaxTime = 0.0;
        double programKernelsMaxTime = 0.0;
        double programVersionTotalTime = 0.0;
        double programKernelsTotalTime = 0.0;
    };

    ProgramDesc applyForcedCompilerFlags(ProgramDesc desc) const;
    void registerProgramForReload(Program* program);
    void unregisterProgramForReload(Program* program);

    ref<const ProgramVersion> createProgramVersion(const Program& program, std::string& log) const;

    ref<const ProgramKernels> createProgramKernels(
        const Program& program,
        const ProgramVersion& programVersion,
        const ProgramVars& programVars,
        std::string& log
    ) const;

    ref<const EntryPointGroupKernels> createEntryPointGroupKernels(
        const std::vector<ref<EntryPointKernel>>& kernels,
        const ref<EntryPointBaseReflection>& pReflector
    ) const;

    /// Get the global HLSL language prelude.
    std::string getHlslLanguagePrelude() const;

    /// Set the global HLSL language prelude.
    void setHlslLanguagePrelude(const std::string& prelude);

    /**
     * Reload and relink all programs.
     * @param[in] forceReload Force reloading all programs.
     * @return True if any program was reloaded, false otherwise.
     */
    bool reloadAllPrograms(bool forceReload = false);

    /**
     * Add a list of defines applied to all programs.
     * @param[in] defineList List of macro definitions.
     */
    void addGlobalDefines(const DefineList& defineList);

    /**
     * Remove a list of defines applied to all programs.
     * @param[in] defineList List of macro definitions.
     */
    void removeGlobalDefines(const DefineList& defineList);

    /**
     * Set compiler arguments applied to all programs.
     * @param[in] args Compiler arguments.
     */
    void setGlobalCompilerArguments(const std::vector<std::string>& args) { mGlobalCompilerArguments = args; }

    /**
     * Get compiler arguments applied to all programs.
     * @return List of compiler arguments.
     */
    const std::vector<std::string>& getGlobalCompilerArguments() const { return mGlobalCompilerArguments; }

    /**
     * Enable/disable global generation of shader debug info.
     * @param[in] enabled Enable/disable.
     */
    void setGenerateDebugInfoEnabled(bool enabled);

    /**
     * Check if global generation of shader debug info is enabled.
     * @return Returns true if enabled.
     */
    bool isGenerateDebugInfoEnabled();

    /**
     * Sets compiler flags that will always be forced on and forced off on each program.
     * If a flag is in both groups, it results in being forced on.
     * @param[in] forceOn Flags to be forced on.
     * @param[in] forceOff Flags to be forced off.
     */
    void setForcedCompilerFlags(ForcedCompilerFlags forcedCompilerFlags);

    /**
     * Retrieve compiler flags that are always forced on all shaders.
     * @return The forced compiler flags.
     */
    ForcedCompilerFlags getForcedCompilerFlags();

    const CompilationStats& getCompilationStats() { return mCompilationStats; }
    void resetCompilationStats() { mCompilationStats = {}; }

private:
    SlangCompileRequest* createSlangCompileRequest(const Program& program) const;

    Device* mpDevice;

    std::vector<Program*> mLoadedPrograms;
    mutable CompilationStats mCompilationStats;

    DefineList mGlobalDefineList;
    std::vector<std::string> mGlobalCompilerArguments;
    bool mGenerateDebugInfo = false;
    ForcedCompilerFlags mForcedCompilerFlags;

    mutable uint32_t mHitGroupID = 0;
};

} // namespace Falcor
