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

#include "BrickedGrid.h"
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Utils/Math/AABB.h"
#include "Utils/Math/Matrix.h"
#include "Utils/UI/Gui.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244 4267)
#endif
#include <nanovdb/NanoVDB.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/HostBuffer.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <filesystem>
#include <memory>
#include <string>

namespace Falcor
{
    struct ShaderVar;

    /** Voxel grid based on NanoVDB.
    */
    class FALCOR_API Grid
    {
    public:
        using SharedPtr = std::shared_ptr<Grid>;

        /** Create a sphere voxel grid.
            \param[in] radius Radius of the sphere in world units.
            \param[in] voxelSize Size of a voxel in world units.
            \param[in] blendRange Range in voxels to blend from 0 to 1 (starting at surface inwards).
            \return A new grid.
        */
        static SharedPtr createSphere(float radius, float voxelSize, float blendRange = 2.f);

        /** Create a box voxel grid.
            \param[in] width Width of the box in world units.
            \param[in] height Height of the box in world units.
            \param[in] depth Depth of the box in world units.
            \param[in] voxelSize Size of a voxel in world units.
            \param[in] blendRange Range in voxels to blend from 0 to 1 (starting at surface inwards).
            \return A new grid.
        */
        static SharedPtr createBox(float width, float height, float depth, float voxelSize, float blendRange = 2.f);

        /** Create a grid from a file.
            Currently only OpenVDB and NanoVDB grids of type float are supported.
            \param[in] path File path of the grid. Can also include a full path or relative path from a data directory.
            \param[in] gridname Name of the grid to load.
            \return A new grid, or nullptr if the grid failed to load.
        */
        static SharedPtr createFromFile(const std::filesystem::path& path, const std::string& gridname);

        /** Render the UI.
        */
        void renderUI(Gui::Widgets& widget);

        /** Bind the grid to a given shader var.
            \param[in] var The shader variable to set the data into.
        */
        void setShaderData(const ShaderVar& var);

        /** Get the minimum index stored in the grid.
        */
        int3 getMinIndex() const;

        /** Get the maximum index stored in the grid.
        */
        int3 getMaxIndex() const;

        /** Get the minimum value stored in the grid.
        */
        float getMinValue() const;

        /** Get the maximum value stored in the grid.
        */
        float getMaxValue() const;

        /** Get the total number of active voxels in the grid.
        */
        uint64_t getVoxelCount() const;

        /** Get the size of the grid in bytes as allocated in GPU memory.
        */
        uint64_t getGridSizeInBytes() const;

        /** Get the grid's bounds in world space.
        */
        AABB getWorldBounds() const;

        /** Get a value stored in the grid.
            Note: This function is not safe for access from multiple threads.
            \param[in] ijk The index-space position to access the data from.
        */
        float getValue(const int3& ijk) const;

        /** Get the raw NanoVDB grid handle.
        */
        const nanovdb::GridHandle<nanovdb::HostBuffer>& getGridHandle() const;

        /** Get the (affine) NanoVDB transformation matrix.
        */
        rmcv::mat4 getTransform() const;

        /** Get the inverse (affine) NanoVDB transformation matrix.
        */
        rmcv::mat4 getInvTransform() const;

    private:
        Grid(nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle);

        static SharedPtr createFromNanoVDBFile(const std::filesystem::path& path, const std::string& gridname);
        static SharedPtr createFromOpenVDBFile(const std::filesystem::path& path, const std::string& gridname);

        // Host data.
        nanovdb::GridHandle<nanovdb::HostBuffer> mGridHandle;
        nanovdb::FloatGrid* mpFloatGrid;
        nanovdb::FloatGrid::AccessorType mAccessor;
        // Device data.
        Buffer::SharedPtr mpBuffer;
        BrickedGrid mBrickedGrid;

        friend class SceneCache;
    };
}
