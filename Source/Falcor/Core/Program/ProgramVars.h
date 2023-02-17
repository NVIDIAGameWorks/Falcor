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
#include "RtBindingTable.h"
#include "Core/Macros.h"
#include "Core/API/fwd.h"
#include "Core/API/ParameterBlock.h"
#include "Core/API/ShaderTable.h"
#include <memory>
#include <vector>

namespace Falcor
{
class GraphicsProgram;
class ComputeProgram;
class ComputeContext;

/**
 * This class manages a program's reflection and variable assignment.
 * It's a high-level abstraction of variables-related concepts such as CBs, texture and sampler assignments, root-signature, descriptor
 * tables, etc.
 */
class FALCOR_API ProgramVars : public ParameterBlock
{
public:
    using SharedPtr = ParameterBlockSharedPtr<ProgramVars>;

    /**
     * Get the program reflection interface
     */
    const ProgramReflection::SharedConstPtr& getReflection() const { return mpReflector; }

protected:
    ProgramVars(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector);

    ProgramReflection::SharedConstPtr mpReflector;
};

class FALCOR_API GraphicsVars : public ProgramVars
{
public:
    using SharedPtr = ParameterBlockSharedPtr<GraphicsVars>;

    /**
     * Create a new graphics vars object.
     * @param[in] pDevice GPU device.
     * @param[in] pReflector A program reflection object containing the requested declarations.
     * @return A new object, or an exception is thrown if creation failed.
     */
    static SharedPtr create(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector);

    /**
     * Create a new graphics vars object.
     * @param[in] pDevice GPU device.
     * @param[in] pProg A program containing the requested declarations. The active version of the program is used.
     * @return A new object, or an exception is thrown if creation failed.
     */
    static SharedPtr create(std::shared_ptr<Device> pDevice, const GraphicsProgram* pProg);

protected:
    GraphicsVars(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector);
};

class FALCOR_API ComputeVars : public ProgramVars
{
public:
    using SharedPtr = ParameterBlockSharedPtr<ComputeVars>;

    /**
     * Create a new compute vars object.
     * @param[in] pDevice GPU device.
     * @param[in] pReflector A program reflection object containing the requested declarations.
     * @return A new object, or an exception is thrown if creation failed.
     */
    static SharedPtr create(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector);

    /**
     * Create a new compute vars object.
     * @param[in] pDevice GPU device.
     * @param[in] pProg A program containing the requested declarations. The active version of the program is used.
     * @return A new object, or an exception is thrown if creation failed.
     */
    static SharedPtr create(std::shared_ptr<Device> pDevice, const ComputeProgram* pProg);

    /**
     * Dispatch the program using the argument values set in this object.
     */
    void dispatchCompute(ComputeContext* pContext, uint3 const& threadGroupCount);

protected:
    ComputeVars(std::shared_ptr<Device> pDevice, const ProgramReflection::SharedConstPtr& pReflector);
};

class RtStateObject;

/**
 * This class manages a raytracing program's reflection and variable assignment.
 */
class FALCOR_API RtProgramVars : public ProgramVars
{
public:
    using SharedPtr = ParameterBlockSharedPtr<RtProgramVars>;

    /**
     * Create a new ray tracing vars object.
     * @param[in] pDevice GPU device.
     * @param[in] pProgram The ray tracing program.
     * @param[in] pBindingTable The raytracing binding table.
     * @return A new object, or an exception is thrown if creation failed.
     */
    static SharedPtr create(
        std::shared_ptr<Device> pDevice,
        const RtProgram::SharedPtr& pProgram,
        const RtBindingTable::SharedPtr& pBindingTable
    );

    bool prepareShaderTable(RenderContext* pCtx, RtStateObject* pRtso);

    ShaderTablePtr getShaderTable() const { return mpShaderTable; }
    uint32_t getMissVarsCount() const { return uint32_t(mMissVars.size()); }
    uint32_t getTotalHitVarsCount() const { return uint32_t(mHitVars.size()); }
    uint32_t getRayTypeCount() const { return mRayTypeCount; }
    uint32_t getGeometryCount() const { return mGeometryCount; }

    const std::vector<int32_t>& getUniqueEntryPointGroupIndices() const { return mUniqueEntryPointGroupIndices; }

private:
    struct EntryPointGroupInfo
    {
        int32_t entryPointGroupIndex = -1;
    };

    using VarsVector = std::vector<EntryPointGroupInfo>;

    RtProgramVars(std::shared_ptr<Device> pDevice, const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable);

    void init(const RtBindingTable::SharedPtr& pBindingTable);

    uint32_t mRayTypeCount = 0;                         ///< Number of ray types (= number of hit groups per geometry).
    uint32_t mGeometryCount = 0;                        ///< Number of geometries.
    std::vector<int32_t> mUniqueEntryPointGroupIndices; ///< Indices of all unique entry point groups that we use in the associated program.

    mutable ShaderTablePtr mpShaderTable;                    ///< GPU shader table.
    mutable RtStateObject* mpCurrentRtStateObject = nullptr; ///< The RtStateObject used to create the current shader table.

    VarsVector mRayGenVars;
    VarsVector mMissVars;
    VarsVector mHitVars;
};

} // namespace Falcor
