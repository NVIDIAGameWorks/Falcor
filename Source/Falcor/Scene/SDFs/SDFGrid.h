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
#include "Core/API/Texture.h"
#include "Scene/SDFs/SDF3DPrimitiveCommon.slang"
#include "RenderGraph/BasePasses/ComputePass.h"
#include <memory>
#include <vector>
#include <utility>
#include <filesystem>
#include <unordered_map>

namespace Falcor
{
    class RenderContext;
    struct ShaderVar;

    /** SDF grid base class, stored by distance values at grid cell/voxel corners.
        The local space of the SDF grid is [-0.5, 0.5]^3 meaning that initial distances used to create the SDF grid should be within the range of [-sqrt(3), sqrt(3)].

        SDF grid implementations create AABBs that should be used as procedural primitives to create acceleration structures.
        SDF grids can currently not be rendered using rasterization.
        Instead, SDF grids must be built into an acceleration structure and may then be ray traced using an intersection shader or inline intersection test.

        There are four different SDF grid implementations:
        1.  The "Normalized Dense Grid" (NDSDFGrid) creates a dense hierarchy of volume textures (they are not mip-levels as they are all have widths of (2^l) - 1.
            Values stored in these volume textures are normalized to be in the range [-1, 1] where 1 represents whatever narrow band thickness is chosen half voxel diagonals.
            For example: Using a narrow band thickness of 3, a value of 1 in the NDSDFGrid will represent a grid distance of 1.5 voxel diagonals.

        2.  The "SDF Sparse Voxel Set" (SDFSVS) creates a sparse set of voxels. Only voxels that overlap the implicit surface formed from the SDF are instantiated.
            The SDFSVS then creates AABBs for all of these voxels, this buffer of AABBs is suitable for use as procedural geometry when creating a BLAS.
            Distance values in the SDFSVS are normalized to the range [-1, 1] so that a value of 1 represents half of a voxel diagonal.

        3.  The "SDF Sparse Brick Set" (SDFSBS) creates a sparse set of NxNxN bricks, where N can be any number of voxels. Each brick is therefore a dense collection
            of voxels with normalized distances at corners. The distances are normalized to the range [-1, 1] so that a value of 1 represents half of a voxel diagonal.
            If (N+1) is a multiple of 4 then lossy compression can be enabled for the SDFSBS.

        4.  The "SDF Sparse Voxel Octree" (SDFSVO) creates a sparse set of voxels and construct an octree out of them. It is then possible to build a BLAS
            out the AABB buffer constructed by the SDFSVO. This can then be used to intersect rays against the SDFSVO.
            Distances stored in the voxels of the octree are normalized to the range [-1, 1] so that a value of 1 represents half of a voxel diagonal.
    */
    class FALCOR_API SDFGrid
    {
    public:
        using SharedPtr = std::shared_ptr<SDFGrid>;

        enum class Type
        {
            None = 0,
            NormalizedDenseGrid = 1,
            SparseVoxelSet = 2,
            SparseBrickSet = 3,
            SparseVoxelOctree = 4,
        };

        /** Flags indicating if and what was updated in the SDF grid.
        */
        enum class UpdateFlags : uint32_t
        {
            None = 0x0,                 ///< Nothing happened
            AABBsChanged = 0x1,         ///< AABBs changed, requires a BLAS update.
            BuffersReallocated = 0x2,   ///< Buffers were reallocated requiring them to be rebound.

            All = AABBsChanged | BuffersReallocated,
        };

        virtual ~SDFGrid() = default;

        /** Set SDF primitives to be used to construct the SDF grid.
            \param[in] primitives The SDF primitives that define the SDF grid.
            \param[in] gridWidth The targeted width of the SDF grid, the resulting grid may have a larger width.
            \return The primitive ID assigned to primitives[0], consecutive primitives are assigned consecutive primitive IDs.
        */
        uint32_t setPrimitives(const std::vector<SDF3DPrimitive>& primitives, uint32_t gridWidth);

        /** Adds SDF primitives to be used to construct the SDF grid.
            \param[in] primitives The SDF primitives that define the SDF grid.
            \return The primitive ID assigned to primitives[0], consecutive primitives are assigned consecutive primitive IDs.
        */
        uint32_t addPrimitives(const std::vector<SDF3DPrimitive>& primitives);

        /** Remove SDF primitives.
            \param[in] primitiveIDs Primitive IDs to remove from the SDF grid.
        */
        void removePrimitives(const std::vector<uint32_t>& primitiveIDs);

        /** Updates the specified SDF primitives.
            \param[in] primitives A vector of primitive IDs and primitives.
        */
        void updatePrimitives(const std::vector<std::pair<uint32_t, SDF3DPrimitive>>& primitives);

        /** Set the signed distance values of the SDF grid, values are expected to be at the corners of voxels.
            \param[in] cornerValues The corner values for all voxels in the grid.
            \param[in] gridWidth The grid width, note that this represents the grid width in voxels, not in values, i.e., cornerValues should have a size of (gridWidth + 1)^3.
        */
        void setValues(const std::vector<float>& cornerValues, uint32_t gridWidth);

        /** Set the signed distance values of the SDF grid from a file.
            \param[in] path The path of a .sdfg file.
            \return true if the values could be set, otherwise false.
        */
        bool loadValuesFromFile(const std::filesystem::path& path);

        /** Set the signed distance values of the SDF grid to represent a swiss cheese like shape.
            \param[in] gridWidth The grid width, note that this represents the grid width in voxels, not in values, i.e., cornerValues should have a size of (gridWidth + 1)^3.
            \param[in] seed Set the seed used to create the random holes in the swiss cheese..
        */
        void generateCheeseValues(uint32_t gridWidth, uint32_t seed);

        /** Evaluates the SDF grid primitives on to a grid and writes the grid to a file.
            \param[in] path A path to the file that should store the values.
            \return true if the values could be written, otherwise false.
        */
        bool writeValuesFromPrimitivesToFile(const std::filesystem::path& path, RenderContext* pRenderContext = nullptr);

        /** Reads primitives from file and initializes the SDF grid.
            \param[in] path The path to the input file.
            \param[in] gridWidth The targeted width of the SDF grid, the resulting grid may have a larger width.
            \param[in] dir A directory path, if this is empty, the file will be searched for in data directories.
            \return The number of primitives loaded.
        */
        uint32_t loadPrimitivesFromFile(const std::filesystem::path& path, uint32_t gridWidth, const std::filesystem::path& dir);

        /** Write the primitives to file.
            \param[in] path The path to the output file.
            return true if the primitives could be successfully written to file.
        */
        bool writePrimitivesToFile(const std::filesystem::path& path);

        /** Updates the SDF grid and applies changes to it.
            \return A combination of flags that signify what changes were made to the SDF grid.
        */
        virtual UpdateFlags update(RenderContext* pRenderContext) { return UpdateFlags::None; };

        /** Get the name of the SDF grid.
            \return Returns the name.
        */
        const std::string& getName() const { return mName; }

        /** Set the name of the SDF grid.
            \param[in] name Name to set.
        */
        void setName(const std::string& name) { mName = name; }

        /** Returns the width of the grid in voxels.
        */
        uint32_t getGridWidth() const { return mGridWidth; }

        /** Returns the number of primitives in the SDF grid.
        */
        uint32_t getPrimitiveCount() const { return (uint32_t)mPrimitives.size(); }

        /** Returns the primitive corresponding to the given primitiveID.
        */
        const SDF3DPrimitive& getPrimitive(uint32_t primitiveID) const;

        /** Returns the byte size of the SDF grid.
        */
        virtual size_t getSize() const = 0;

        /** Returns the maximum number of bits that could be stored in the primitive ID field of HitInfo.
        */
        virtual uint32_t getMaxPrimitiveIDBits() const = 0;

        /** Returns the type of this SDF grid.
        */
        virtual Type getType() const = 0;

        /** Creates the GPU data structures required to render the SDF grid.
        */
        virtual void createResources(RenderContext* pRenderContext = nullptr, bool deleteScratchData = true) = 0;

        /** Returns an AABB buffer that can be used to create an accelerations strucure using this SDF grid.
        */
        virtual const Buffer::SharedPtr& getAABBBuffer() const = 0;

        /** Return the number of AABBs used to create this SDF grid.
        */
        virtual uint32_t getAABBCount() const = 0;

        /** Binds the SDF grid into a given shader var.
        */
        virtual void setShaderData(const ShaderVar& var) const = 0;

        /** Return the scaling factor that represent how the grid resolution has changed from when it was loaded. The resolution can change if loaded by the SBS grid and then edited by the SDFEditor.
        */
        virtual float getResolutionScalingFactor() const { return 1.0f; };

        /** Sets the resolution scaling factor to 1.0.
        */
        virtual void resetResolutionScalingFactor() {};

        /** Bake primitives into the grid representation (sdfg-file).
            \param[in] batchSize The number of primitives to bake.
        */
        void bakePrimitives(uint32_t batchSize);

        /** Check if the grid was initialized with primitives.
            \return True if the grid was initialized with primitives, else false.
        */
        bool wasInitializedWithPrimitives() const { return mInitializedWithPrimitives; };

        /** Get the number of primitives that have been baked into the grid representation (sdfg-file).
        */
        uint32_t getBakedPrimitiveCount() const { return mBakedPrimitiveCount; };

        static std::string getTypeName(Type type);

    protected:
        virtual void setValuesInternal(const std::vector<float>& cornerValues) = 0;

        void createEvaluatePrimitivesPass(bool writeToTexture3D, bool mergeWithSDField);

        void updatePrimitivesBuffer();

        std::string             mName;
        uint32_t                mGridWidth = 0;

        // Primitive data.
        std::vector<SDF3DPrimitive> mPrimitives;
        std::unordered_map<uint32_t, uint32_t> mPrimitiveIDToIndex;
        uint32_t                mNextPrimitiveID = 0;
        bool                    mPrimitivesDirty = false;           ///< True if the primitives have changed.
        Buffer::SharedPtr       mpPrimitivesBuffer;                 ///< Holds the primitives that should be rendered.
        uint32_t                mPrimitivesExcludedFromBuffer = 0;  ///< Number of primitives to exclude from the primitive buffer.
        uint32_t                mBakedPrimitiveCount = 0;           ///< Number of primitives that will be baked into the value representation.
        bool                    mBakePrimitives = false;            ///< True if the primitives should be baked into the value representation.
        bool                    mHasGridRepresentation = false;     ///< True if a value representation exists.
        bool                    mInitializedWithPrimitives = false; ///< True if the grid was initialized with primitives.

        Texture::SharedPtr      mpSDFGridTexture;                   ///< A texture on the GPU holding the value representation.
        ComputePass::SharedPtr  mpEvaluatePrimitivesPass;
    };

    FALCOR_ENUM_CLASS_OPERATORS(SDFGrid::UpdateFlags);
}
