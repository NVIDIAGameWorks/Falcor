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
#include "stdafx.h"
#include "Grid.h"
#pragma warning(disable:4146 4244 4267 4275 4996)
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridStats.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <openvdb/openvdb.h>
#pragma warning(default:4146 4244 4267 4275 4996)

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

    Grid::SharedPtr Grid::createFromFile(const std::string& filename, const std::string& gridname)
    {
        std::string fullpath;
        if (!findFileInDataDirectories(filename, fullpath))
        {
            logWarning("Error when loading grid. Can't find grid file '" + filename + "'");
            return nullptr;
        }

        auto ext = getExtensionFromFile(fullpath);
        if (ext == "nvdb")
        {
            return createFromNanoVDBFile(fullpath, gridname);
        }
        else if (ext == "vdb")
        {
            return createFromOpenVDBFile(fullpath, gridname);
        }
        else
        {
            logWarning("Error when loading grid. Unsupported grid file '" + filename + "'");
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
    }

    int3 Grid::getMinIndex() const
    {
        return cast(mpFloatGrid->indexBBox().min());
    }

    int3 Grid::getMaxIndex() const
    {
        return cast(mpFloatGrid->indexBBox().max());
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
        return mpBuffer ? mpBuffer->getSize() : (uint64_t)0;
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

    Grid::Grid(nanovdb::GridHandle<nanovdb::HostBuffer> gridHandle)
        : mGridHandle(std::move(gridHandle))
        , mpFloatGrid(mGridHandle.grid<float>())
        , mAccessor(mpFloatGrid->getAccessor())
    {
        if (!mpFloatGrid->hasMinMax())
        {
            nanovdb::gridStats(*mpFloatGrid);
        }

        mpBuffer = Buffer::createStructured(
            sizeof(uint32_t),
            uint32_t(div_round_up(mGridHandle.size(), sizeof(uint32_t))),
            ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource,
            Buffer::CpuAccess::None,
            mGridHandle.data()
        );
    }

    Grid::SharedPtr Grid::createFromNanoVDBFile(const std::string& path, const std::string& gridname)
    {
        if (!nanovdb::io::hasGrid(path, gridname))
        {
            logWarning("Error when loading grid. Can't find grid '" + gridname + "' in '" + path + "'");
            return nullptr;
        }

        auto handle = nanovdb::io::readGrid(path, gridname);
        if (!handle)
        {
            logWarning("Error when loading grid.");
            return nullptr;
        }

        auto floatGrid = handle.grid<float>();
        if (!floatGrid || floatGrid->gridType() != nanovdb::GridType::Float)
        {
            logWarning("Error when loading grid. Grid '" + gridname + "' in '" + path + "' is not of type float");
            return nullptr;
        }

        return SharedPtr(new Grid(std::move(handle)));
    }

    Grid::SharedPtr Grid::createFromOpenVDBFile(const std::string& path, const std::string& gridname)
    {
        openvdb::initialize();

        openvdb::io::File file(path);
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
            logWarning("Error when loading grid. Can't find grid '" + gridname + "' in '" + path + "'");
            return nullptr;
        }

        if (!baseGrid->isType<openvdb::FloatGrid>())
        {
            logWarning("Error when loading grid. Grid '" + gridname + "' in '" + path + "' is not of type float");
            return nullptr;
        }

        openvdb::FloatGrid::Ptr floatGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);
        auto handle = nanovdb::openToNanoVDB(floatGrid);

        return SharedPtr(new Grid(std::move(handle)));
    }


    SCRIPT_BINDING(Grid)
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
        grid.def_static("createFromFile", &Grid::createFromFile, "filename"_a, "gridname"_a);
    }
}
