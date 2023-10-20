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
#include "Grid.h"
#include "GridConverter.h"
#include "Core/API/Device.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/StringUtils.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/Vector.h"
#include "Utils/Math/Matrix.h"
#include "GlobalState.h"
#include "Utils/PathResolving.h"

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4146 4244 4267 4275 4996 4456)
#endif
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridStats.h>
// TODO: GridBuilder.h uses the std::result_of type trait which is deprecated in C++17 and
// removed in C++20. This is an ugly workaround to use C++20's invoke_result type trait.
// This really should be fixed in nanovdb instead!
#define result_of invoke_result
#include <nanovdb/util/GridBuilder.h>
#undef result_of
#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/openvdb.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif

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

    ref<Grid> Grid::createSphere(ref<Device> pDevice, float radius, float voxelSize, float blendRange)
    {
        auto handle = nanovdb::createFogVolumeSphere<float>(radius, nanovdb::Vec3f(0.f), voxelSize, blendRange);
        return ref<Grid>(new Grid(pDevice, std::move(handle)));
    }

    ref<Grid> Grid::createBox(ref<Device> pDevice, float width, float height, float depth, float voxelSize, float blendRange)
    {
        auto handle = nanovdb::createFogVolumeBox<float>(width, height, depth, nanovdb::Vec3f(0.f), voxelSize, blendRange);
        return ref<Grid>(new Grid(pDevice, std::move(handle)));
    }

    ref<Grid> Grid::createFromFile(ref<Device> pDevice, const std::filesystem::path& path, const std::string& gridname)
    {
        if (!std::filesystem::exists(path))
        {
            logWarning("Error when loading grid. Can't open grid file '{}'.", path);
            return nullptr;
        }

        if (hasExtension(path, "nvdb"))
        {
            return createFromNanoVDBFile(pDevice, path, gridname);
        }
        else if (hasExtension(path, "vdb"))
        {
            return createFromOpenVDBFile(pDevice, path, gridname);
        }
        else
        {
            logWarning("Error when loading grid. Unsupported grid file '{}'.", path);
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

    void Grid::bindShaderData(const ShaderVar& var)
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
        return mpFloatGrid->tree().root().minimum();
    }

    float Grid::getMaxValue() const
    {
        return mpFloatGrid->tree().root().maximum();
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

    float4x4 Grid::getTransform() const
    {
        const auto& gridMap = mGridHandle.gridMetaData()->map();
        const float3x3 affine = math::matrixFromCoefficients<float, 3, 3>(gridMap.mMatF);
        const float3 translation = float3(gridMap.mVecF[0], gridMap.mVecF[1], gridMap.mVecF[2]);
        return math::translate(float4x4(affine), translation);
    }

    float4x4 Grid::getInvTransform() const
    {
        const auto& gridMap = mGridHandle.gridMetaData()->map();
        const float3x3 invAffine = math::matrixFromCoefficients<float, 3, 3>(gridMap.mInvMatF);
        const float3 translation = float3(gridMap.mVecF[0], gridMap.mVecF[1], gridMap.mVecF[2]);
        return math::translate(float4x4(invAffine), -translation);
    }

    Grid::Grid(ref<Device> pDevice, nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle)
        : mpDevice(pDevice)
        , mGridHandle(std::move(gridHandle))
        , mpFloatGrid(mGridHandle.grid<float>())
        , mAccessor(mpFloatGrid->getAccessor())
    {
        if (!mpFloatGrid->hasMinMax())
        {
            nanovdb::gridStats(*mpFloatGrid);
        }

        // Keep both NanoVDB and brick textures resident in GPU memory for simplicity for now (~15% increased footprint).
        mpBuffer = mpDevice->createStructuredBuffer(
            sizeof(uint32_t),
            uint32_t(div_round_up(mGridHandle.size(), sizeof(uint32_t))),
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            MemoryType::DeviceLocal,
            mGridHandle.data()
        );
        using NanoVDBGridConverter = NanoVDBConverterBC4;
        mBrickedGrid = NanoVDBGridConverter(mpFloatGrid).convert(mpDevice);
    }

    ref<Grid> Grid::createFromNanoVDBFile(ref<Device> pDevice, const std::filesystem::path& path, const std::string& gridname)
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

        return ref<Grid>(new Grid(pDevice, std::move(handle)));
    }

    ref<Grid> Grid::createFromOpenVDBFile(ref<Device> pDevice, const std::filesystem::path& path, const std::string& gridname)
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

        return ref<Grid>(new Grid(pDevice, std::move(handle)));
    }


    FALCOR_SCRIPT_BINDING(Grid)
    {
        using namespace pybind11::literals;

        pybind11::class_<Grid, ref<Grid>> grid(m, "Grid");
        grid.def_property_readonly("voxelCount", &Grid::getVoxelCount);
        grid.def_property_readonly("minIndex", &Grid::getMinIndex);
        grid.def_property_readonly("maxIndex", &Grid::getMaxIndex);
        grid.def_property_readonly("minValue", &Grid::getMinValue);
        grid.def_property_readonly("maxValue", &Grid::getMaxValue);

        grid.def("getValue", &Grid::getValue, "ijk"_a);

        auto createSphere = [] (float radius, float voxelSize, float blendRange)
        {
            return Grid::createSphere(accessActivePythonSceneBuilder().getDevice(), radius, voxelSize, blendRange);
        };
        grid.def_static("createSphere", createSphere, "radius"_a, "voxelSize"_a, "blendRange"_a = 3.f); // PYTHONDEPRECATED

        auto createBox = [] (float width, float height, float depth, float voxelSize, float blendRange)
        {
            return Grid::createBox(accessActivePythonSceneBuilder().getDevice(), width, height, depth, voxelSize, blendRange);
        };
        grid.def_static("createBox", createBox, "width"_a, "height"_a, "depth"_a, "voxelSize"_a, "blendRange"_a = 3.f); // PYTHONDEPRECATED

        auto createFromFile = [] (const std::filesystem::path& path, const std::string& gridname)
        {
            return Grid::createFromFile(accessActivePythonSceneBuilder().getDevice(), getActiveAssetResolver().resolvePath(path), gridname);
        };
        grid.def_static("createFromFile", createFromFile, "path"_a, "gridname"_a); // PYTHONDEPRECATED
    }
}
