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
#include "Grid.h"
#pragma warning(push)
#pragma warning(disable : 4146 4244 4267 4275 4996)
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/openvdb.h>
#pragma warning(pop)
#include <glm/gtc/type_ptr.hpp>
#include "GridConverter.h"


namespace Falcor
{
    namespace
    {
        float3 cast(const nanovdb::Vec3f& v)
        {
            return float3(v[0], v[1], v[2]);
        }

        float3 cast(const nanovdb::Vec3R& v)
        {
            return float3(v[0], v[1], v[2]);
        }

        int3 cast(const nanovdb::Coord& c)
        {
            return int3(c[0], c[1], c[2]);
        }
    }

    Grid::SharedPtr Grid::createSphere(float radius, float voxelSize, float blendRange)
    {
        auto handle = nanovdb::createFogVolumeSphere(radius, nanovdb::Vec3R(0.0), voxelSize, blendRange);
        return SharedPtr(new Grid(std::move(handle)));
    }

    Grid::SharedPtr Grid::createBox(float width, float height, float depth, float voxelSize, float blendRange)
    {
        auto handle = nanovdb::createFogVolumeBox(width, height, depth, nanovdb::Vec3R(0.0), voxelSize, blendRange);
        return SharedPtr(new Grid(std::move(handle)));
    }

    Grid::SharedPtr Grid::createFromFile(const std::filesystem::path& path, const std::string& gridname)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            logWarning("Error when loading grid. Can't find grid file '{}'.", path);
            return nullptr;
        }

        if (hasExtension(fullPath, "nvdb"))
        {
            return createFromNanoVDBFile(fullPath, gridname);
        }
        else if (hasExtension(fullPath, "vdb"))
        {
            return createFromOpenVDBFile(fullPath, gridname);
        }
        else
        {
            logWarning("Error when loading grid. Unsupported grid file '{}'.", fullPath);
            return nullptr;
        }
    }

    void Grid::renderUI(Gui::Widgets& widget)
    {
        std::ostringstream oss;
        oss << "Voxel count: " << getVoxelCount() << std::endl
            << "Minimum index: " << to_string(getMinIndex()) << std::endl
            << "Maximum index: " << to_string(getMaxIndex()) << std::endl
            << "Minimum value: " << getMinValue() << std::endl
            << "Maximum value: " << getMaxValue() << std::endl
            << "Memory: " << formatByteSize(getGridSizeInBytes()) << std::endl;
        widget.text(oss.str());
    }

    void Grid::setShaderData(const ShaderVar& var)
    {
        var["buf"] = mpBuffer;
        var["rangeTex"] = mBrickedGrid.range;
        var["indirectionTex"] = mBrickedGrid.indirection;
        var["atlasTex"] = mBrickedGrid.atlas;
        var["minIndex"] = getMinIndex();
        var["minValue"] = getMinValue();
        var["maxIndex"] = getMaxIndex();
        var["maxValue"] = getMaxValue();
    }

    int3 Grid::getMinIndex() const
    {
        return cast(mpFloatGrid->indexBBox().min()) & (~7); // The volume texture path requires the index bounding box to fall on a brick boundary (multiple of 8).
    }

    int3 Grid::getMaxIndex() const
    {
        return (cast(mpFloatGrid->indexBBox().max()) + 7) & (~7); // The volume texture path requires the index bounding box to fall on a brick boundary (multiple of 8).
    }

    float Grid::getMinValue() const
    {
        return mpFloatGrid->tree().root().valueMin();
    }

    float Grid::getMaxValue() const
    {
        return mpFloatGrid->tree().root().valueMax();
    }

    uint64_t Grid::getVoxelCount() const
    {
        return mpFloatGrid->activeVoxelCount();
    }

    uint64_t Grid::getGridSizeInBytes() const
    {
        const uint64_t nvdb = mpBuffer ? mpBuffer->getSize() : (uint64_t)0;
        const uint64_t bricks = (mBrickedGrid.range ? mBrickedGrid.range->getTextureSizeInBytes() : (uint64_t)0) +
            (mBrickedGrid.indirection ? mBrickedGrid.indirection->getTextureSizeInBytes() : (uint64_t)0) +
            (mBrickedGrid.atlas ? mBrickedGrid.atlas->getTextureSizeInBytes() : (uint64_t)0);
        return nvdb + bricks;
    }

    AABB Grid::getWorldBounds() const
    {
        auto bounds = mpFloatGrid->worldBBox();
        return AABB(cast(bounds.min()), cast(bounds.max()));
    }

    float Grid::getValue(const int3& ijk) const
    {
        return mAccessor.getValue(nanovdb::Coord(ijk.x, ijk.y, ijk.z));
    }

    const nanovdb::GridHandle<nanovdb::HostBuffer>& Grid::getGridHandle() const
    {
        return mGridHandle;
    }

    glm::mat4 Grid::getTransform() const
    {
        const auto& gridMap = mGridHandle.gridMetaData()->map();
        const float3x3 affine = glm::make_mat3(gridMap.mMatF);
        const float3 translation = float3(gridMap.mVecF[0], gridMap.mVecF[1], gridMap.mVecF[2]);
        return glm::translate(float4x4(affine), translation);
    }

    glm::mat4 Grid::getInvTransform() const
    {
        const auto& gridMap = mGridHandle.gridMetaData()->map();
        const float3x3 invAffine = glm::make_mat3(gridMap.mInvMatF);
        const float3 translation = float3(gridMap.mVecF[0], gridMap.mVecF[1], gridMap.mVecF[2]);
        return glm::translate(float4x4(invAffine), -translation);
    }

    Grid::Grid(nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle)
        : mGridHandle(std::move(gridHandle))
        , mpFloatGrid(mGridHandle.grid<float>())
        , mAccessor(mpFloatGrid->getAccessor())
    {
        if (!mpFloatGrid->hasMinMax())
        {
            nanovdb::gridStats(*mpFloatGrid);
        }

        // Keep both NanoVDB and brick textures resident in GPU memory for simplicity for now (~15% increased footprint).
        mpBuffer = Buffer::createStructured(
            sizeof(uint32_t),
            uint32_t(div_round_up(mGridHandle.size(), sizeof(uint32_t))),
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            Buffer::CpuAccess::None,
            mGridHandle.data()
        );
        using NanoVDBGridConverter = NanoVDBConverterBC4;
        mBrickedGrid = NanoVDBGridConverter(mpFloatGrid).convert();
    }

    Grid::SharedPtr Grid::createFromNanoVDBFile(const std::filesystem::path& path, const std::string& gridname)
    {
        if (!nanovdb::io::hasGrid(path.string(), gridname))
        {
            logWarning("Error when loading grid. Can't find grid '{}' in '{}'.", gridname, path);
            return nullptr;
        }

        auto handle = nanovdb::io::readGrid(path.string(), gridname);
        if (!handle)
        {
            logWarning("Error when loading grid.");
            return nullptr;
        }

        auto floatGrid = handle.grid<float>();
        if (!floatGrid || floatGrid->gridType() != nanovdb::GridType::Float)
        {
            logWarning("Error when loading grid. Grid '{}' in '{}' is not of type float.", gridname, path);
            return nullptr;
        }

        if (floatGrid->isEmpty())
        {
            logWarning("Grid '{}' in '{}' is empty.", gridname, path);
            return nullptr;
        }

        return SharedPtr(new Grid(std::move(handle)));
    }

    Grid::SharedPtr Grid::createFromOpenVDBFile(const std::filesystem::path& path, const std::string& gridname)
    {
        openvdb::initialize();

        openvdb::io::File file(path.string());
        file.open();

        openvdb::GridBase::Ptr baseGrid;
        for (auto it = file.beginName(); it != file.endName(); ++it)
        {
            if (it.gridName() == gridname)
            {
                baseGrid = file.readGrid(it.gridName());
                break;
            }
        }

        file.close();

        if (!baseGrid)
        {
            logWarning("Error when loading grid. Can't find grid '{}' in '{}'.", gridname, path);
            return nullptr;
        }

        if (!baseGrid->isType<openvdb::FloatGrid>())
        {
            logWarning("Error when loading grid. Grid '{}' in '{}' is not of type float.", gridname, path);
            return nullptr;
        }

        if (baseGrid->empty())
        {
            logWarning("Grid '{}' in '{}' is empty.", gridname, path);
            return nullptr;
        }

        openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        auto handle = nanovdb::openToNanoVDB(floatGrid);

        return SharedPtr(new Grid(std::move(handle)));
    }


    FALCOR_SCRIPT_BINDING(Grid)
    {
        pybind11::class_<Grid, Grid::SharedPtr> grid(m, "Grid");
        grid.def_property_readonly("voxelCount", &Grid::getVoxelCount);
        grid.def_property_readonly("minIndex", &Grid::getMinIndex);
        grid.def_property_readonly("maxIndex", &Grid::getMaxIndex);
        grid.def_property_readonly("minValue", &Grid::getMinValue);
        grid.def_property_readonly("maxValue", &Grid::getMaxValue);

        grid.def("getValue", &Grid::getValue, "ijk"_a);

        grid.def_static("createSphere", &Grid::createSphere, "radius"_a, "voxelSize"_a, "blendRange"_a = 3.f);
        grid.def_static("createBox", &Grid::createBox, "width"_a, "height"_a, "depth"_a, "voxelSize"_a, "blendRange"_a = 3.f);
        grid.def_static("createFromFile", &Grid::createFromFile, "path"_a, "gridname"_a);
    }
}
