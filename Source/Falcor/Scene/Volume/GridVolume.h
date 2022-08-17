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
#include "Grid.h"
#include "GridVolumeData.slang"
#include "Core/Macros.h"
#include "Utils/Math/AABB.h"
#include "Utils/Math/Matrix.h"
#include "Utils/UI/Gui.h"
#include "Scene/Animation/Animatable.h"
#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Falcor
{
    /** Describes a grid volume (heterogeneous) in the scene.
        The absorbing/scattering medium is defined by a density voxel grid and additional parameters.
        The emission is defined by an emission voxel grid and additional parameters.
        Grids are stored in grid slots (density, emission) and can either be static, using one grid per slot,
        or dynamic, using a sequence of grids per slot.
    */
    class FALCOR_API GridVolume : public Animatable
    {
    public:
        using SharedPtr = std::shared_ptr<GridVolume>;

        using GridSequence = std::vector<Grid::SharedPtr>;

        /** Flags indicating if and what was updated in the volume.
        */
        enum class UpdateFlags
        {
            None                = 0x0,  ///< Nothing updated.
            PropertiesChanged   = 0x1,  ///< Volume properties changed.
            GridsChanged        = 0x2,  ///< Volume grids changed.
            TransformChanged    = 0x4,  ///< Volume transform changed.
            BoundsChanged       = 0x8,  ///< Volume world-space bounds changed.
        };

        /** Grid slots available in the volume.
        */
        enum class GridSlot
        {
            Density,
            Emission,

            Count // Must be last
        };

        /** Specifies how emission is rendered.
        */
        enum class EmissionMode
        {
            Direct,
            Blackbody,
        };

        /** Create a new volume.
            \param[in] name The volume name.
        */
        static SharedPtr create(const std::string& name);

        /** Render the UI.
            \return True if the volume was modified.
        */
        bool renderUI(Gui::Widgets& widget);

        /** Returns the updates since the last call to clearUpdates.
        */
        UpdateFlags getUpdates() const { return mUpdates; }

        /** Clears the updates.
        */
        void clearUpdates() { mUpdates = UpdateFlags::None; }

        /** Set the volume name.
        */
        void setName(const std::string& name) { mName = name; }

        /** Get the volume name.
        */
        const std::string& getName() const { return mName; }

        /** Load a single grid from a file to a grid slot.
            Note: This will replace any existing grid sequence for that slot with just a single grid.
            \param[in] slot Grid slot.
            \param[in] path File path of the grid. Can also include a full path or relative path from a data directory.
            \param[in] gridname Name of the grid to load.
            \return Returns true if grid was loaded successfully.
        */
        bool loadGrid(GridSlot slot, const std::filesystem::path& path, const std::string& gridname);

        /** Load a sequence of grids from files to a grid slot.
            Note: This will replace any existing grid sequence for that slot.
            \param[in] slot Grid slot.
            \param[in] paths File paths of the grids. Can also include a full path or relative path from a data directory.
            \param[in] gridname Name of the grid to load.
            \param[in] keepEmpty Add empty (nullptr) grids to the sequence if one cannot be loaded from the file.
            \return Returns the length of the loaded sequence.
        */
        uint32_t loadGridSequence(GridSlot slot, const std::vector<std::filesystem::path>& paths, const std::string& gridname, bool keepEmpty = true);

        /** Load a sequence of grids from a directory to a grid slot.
            Note: This will replace any existing grid sequence for that slot.
            \param[in] slot Grid slot.
            \param[in] path Directory containing grid files. Can also include a full path or relative path from a data directory.
            \param[in] gridname Name of the grid to load.
            \param[in] keepEmpty Add empty (nullptr) grids to the sequence if one cannot be loaded from the file.
            \return Returns the length of the loaded sequence.
        */
        uint32_t loadGridSequence(GridSlot slot, const std::filesystem::path& path, const std::string& gridname, bool keepEmpty = true);

        /** Set the grid sequence for the specified slot.
        */
        void setGridSequence(GridSlot slot, const GridSequence& grids);

        /** Get the grid sequence for the specified slot.
        */
        const GridSequence& getGridSequence(GridSlot slot) const;

        /** Set the grid for the specified slot.
            Note: This will replace any existing grid sequence for that slot with just a single grid.
        */
        void setGrid(GridSlot slot, const Grid::SharedPtr& grid);

        /** Get the current grid from the specified slot.
        */
        const Grid::SharedPtr& getGrid(GridSlot slot) const;

        /** Get a list of all grids used for this volume.
        */
        std::vector<Grid::SharedPtr> getAllGrids() const;

        /** Sets the current frame of the grid sequence to use.
        */
        void setGridFrame(uint32_t gridFrame);

        /** Get the current frame of the grid sequence.
        */
        uint32_t getGridFrame() const { return mGridFrame; }

        /** Get the number of frames in the grid sequence.
            Note: This returns 1 even if there are no grids loaded.
        */
        uint32_t getGridFrameCount() const { return mGridFrameCount; }

        /** Set the frame rate for grid playback.
        */
        void setFrameRate(double frameRate);

        /** Get the frame rate for grid playback.
        */
        double getFrameRate() const { return mFrameRate; }

        /** Enable/disable grid playback.
        */
        void setPlaybackEnabled(bool enabled);

        /** Check if grid playback is enabled.
        */
        bool isPlaybackEnabled() const { return mPlaybackEnabled; }

        /** Update the selected grid frame based on global time in seconds.
        */
        void updatePlayback(double curentTime);

        /** Set the density grid.
        */
        void setDensityGrid(const Grid::SharedPtr& densityGrid) { setGrid(GridSlot::Density, densityGrid); };

        /** Get the density grid.
        */
        const Grid::SharedPtr& getDensityGrid() const { return getGrid(GridSlot::Density); }

        /** Set the density scale factor.
        */
        void setDensityScale(float densityScale);

        /** Get the density scale factor.
        */
        float getDensityScale() const { return mData.densityScale; }

        /** Set the emission grid.
        */
        void setEmissionGrid(const Grid::SharedPtr& emissionGrid) { setGrid(GridSlot::Emission, emissionGrid); }

        /** Get the emission grid.
        */
        const Grid::SharedPtr& getEmissionGrid() const { return getGrid(GridSlot::Emission); }

        /** Set the emission scale factor.
        */
        void setEmissionScale(float emissionScale);

        /** Get the emission scale factor.
        */
        float getEmissionScale() const { return mData.emissionScale; }

        /** Set the scattering albedo.
        */
        void setAlbedo(const float3& albedo);

        /** Get the scattering albedo.
        */
        const float3& getAlbedo() const { return mData.albedo; }

        /** Set the phase function anisotropy (forward or backward scattering).
        */
        void setAnisotropy(float anisotropy);

        /** Get the phase function anisotropy.
        */
        float getAnisotropy() const { return mData.anisotropy; }

        /** Set the emission mode.
        */
        void setEmissionMode(EmissionMode emissionMode);

        /** Get the emission mode.
        */
        EmissionMode getEmissionMode() const;

        /** Set the emission base temperature (K).
        */
        void setEmissionTemperature(float emissionTemperature);

        /** Get the emission base temperature (K).
        */
        float getEmissionTemperature() const { return mData.emissionTemperature; }

        /** Returns the grid volume data struct.
        */
        const GridVolumeData& getData() const { return mData; }

        /** Returns the volume bounds in world space.
        */
        const AABB& getBounds() const { return mBounds; }

        void updateFromAnimation(const rmcv::mat4& transform) override;

    private:
        GridVolume(const std::string& name);

        void updateSequence();
        void updateBounds();

        void markUpdates(UpdateFlags updates);
        void setFlags(uint32_t flags);

        std::string mName;
        std::array<GridSequence, (size_t)GridSlot::Count> mGrids;
        uint32_t mGridFrame = 0;
        uint32_t mGridFrameCount = 1;
        double mFrameRate = 30.f;
        bool mPlaybackEnabled = false;
        AABB mBounds;
        GridVolumeData mData;
        mutable UpdateFlags mUpdates = UpdateFlags::None;

        friend class SceneCache;
    };

    FALCOR_ENUM_CLASS_OPERATORS(GridVolume::UpdateFlags);
}
