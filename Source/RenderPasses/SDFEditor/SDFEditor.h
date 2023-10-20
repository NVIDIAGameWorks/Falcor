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
#include "Falcor.h"
#include "Core/Pass/FullScreenPass.h"
#include "RenderGraph/RenderPass.h"
#include "SDFEditorTypes.slang"
#include "Marker2DSet.h"
#include "SelectionWheel.h"

using namespace Falcor;

class SDFEditor : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SDFEditor, "SDFEditor", "Signed distance function (SDF) editor");

    static ref<SDFEditor> create(ref<Device> pDevice, const Properties& props) { return make_ref<SDFEditor>(pDevice, props); }

    SDFEditor(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(RenderContext* pRenderContext, Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;

    static void registerBindings(pybind11::module& m);

private:
    enum class TransformationState
    {
        None,
        Translating,
        Rotating,
        Scaling
    };

    struct KeyboardButtonsPressed
    {
        bool undo = false;
        bool redo = false;
        bool shift = false;
        bool control = false;
        bool prevShift = false;
        bool prevControl = false;

        void registerCurrentStatesIntoPrevious()
        {
            prevShift = shift;
            prevControl = control;
        }
    };

    struct UI2D
    {
        bool recordStartingMousePos = false;
        float scrollDelta = 0.0f;
        KeyboardButtonsPressed keyboardButtonsPressed;
        float2 startMousePosition = {0.0f, 0.0f};
        float2 currentMousePosition = {0.0f, 0.0f};
        float2 prevMousePosition = {0.0f, 0.0f};
        CpuTimer timer;
        CpuTimer::TimePoint timeOfReleaseMainGUIKey;
        bool fadeAwayGUI = false;
        bool drawCurrentModes = true;
        std::unique_ptr<Marker2DSet> pMarker2DSet;
        std::unique_ptr<SelectionWheel> pSelectionWheel;
        float currentBlobbing = 0.0f;
        SDF3DShapeType currentEditingShape = SDF3DShapeType::Sphere;
        SDFOperationType currentEditingOperator = SDFOperationType::Union;
        SDFBBRenderSettings bbRenderSettings;
        SDFGridPlane previousGridPlane;
        SDFGridPlane gridPlane;
        SDFGridPlane previousSymmetryPlane;
        SDFGridPlane symmetryPlane;
    };

    struct CurrentEdit
    {
        uint32_t instanceID;
        SdfGridID gridID;
        ref<SDFGrid> pSDFGrid;
        SDF3DPrimitive primitive;
        SDF3DPrimitive symmetryPrimitive;
        uint32_t primitiveID = UINT32_MAX;
        uint32_t symmetryPrimitiveID = UINT32_MAX;
    };

    struct SDFEdit
    {
        SdfGridID gridID;
        uint32_t primitiveID;
    };

    struct UndoneSDFEdit
    {
        SdfGridID gridID;
        SDF3DPrimitive primitive;
    };

private:
    void bindShaderData(const ShaderVar& var, const ref<Texture>& pInputColor, const ref<Texture>& pVBuffer);
    void fetchPreviousVBufferAndZBuffer(RenderContext* pRenderContext, ref<Texture>& pVBuffer, ref<Texture>& pDepth);

    // 2D GUI functions.
    void setup2DGUI();
    bool isMainGUIKeyDown() const;
    void setupPrimitiveAndOperation(
        const float2& center,
        const float markerSize,
        const SDF3DShapeType editingPrimitive,
        const SDFOperationType editingOperator,
        const float4& color,
        const float alpha = 1.0f
    );
    void setupCurrentModes2D();
    void manipulateGridPlane(
        SDFGridPlane& gridPlane,
        SDFGridPlane& previousGridPlane,
        bool isTranslationKeyDown,
        bool isConstrainedManipulationKeyDown
    );
    void rotateGridPlane(
        const float mouseDiff,
        const float3& rotationVector,
        const float3& inNormal,
        const float3& inRightVector,
        float3& outNormal,
        float3& outRightVector,
        const bool fromPreviousMouse = true
    );
    void translateGridPlane(const float mouseDiff, const float3& translationVector, const float3& inPosition, float3& outPosition);
    bool gridPlaneManipulated() const;
    bool symmetryPlaneManipulated() const;

    // Editing functions
    void updateEditShapeType();
    void updateEditOperationType();
    void updateSymmetryPrimitive();
    void addEditPrimitive(bool addToCurrentEdit, bool addToHistory);
    void removeEditPrimitives();
    void updateEditPrimitives();

    // Input and actions
    void handleActions();
    void handleToggleSymmetryPlane();
    void handleToggleEditing();
    void handleEditMovement();
    void handleAddPrimitive();
    bool handlePicking(const float2& currentMousePos, float3& p);
    uint32_t calcPrimitivesAffectedCount(uint32_t keyPressedCount);
    void handleUndo();
    void handleRedo();

    void bakePrimitives();

private:
    ref<Scene> mpScene;                  ///< The current scene.
    ref<Camera> mpCamera;                ///< The camera.
    ref<FullScreenPass> mpGUIPass;       ///< A full screen pass drawing the 2D GUI.
    ref<Fbo> mpFbo;                      ///< Frame buffer object.
    ref<Texture> mpEditingVBuffer;       ///< A copy of the VBuffer used while moving/adding a primitive.
    ref<Texture> mpEditingLinearZBuffer; ///< A copy of the linear Z buffer used while moving/adding a primitive.
    ref<Buffer> mpSDFEditingDataBuffer;  ///< A buffer that contain current Edit data for GUI visualization.

    struct
    {
        TransformationState prevState = TransformationState::None;
        TransformationState state = TransformationState::None;
        Transform startInstanceTransform;
        Transform startPrimitiveTransform;
        SDF3DPrimitive startPrimitive;
        float3 startPlanePos = {0.0f, 0.0f, 0.0f};
        float3 referencePlaneDir = {0.0f, 0.0f, 0.0f};
        SDFEditorAxis axis = SDFEditorAxis::All;
        SDFEditorAxis prevAxis = SDFEditorAxis::All;
        float2 startMousePos = {0.0f, 0.0f};
    } mPrimitiveTransformationEdit;

    struct
    {
        TransformationState prevState = TransformationState::None;
        TransformationState state = TransformationState::None;
        Transform startTransform;
        float3 startPlanePos = {0.0f, 0.0f, 0.0f};
        float3 referencePlaneDir = {0.0f, 0.0f, 0.0f};
        float prevScrollTotal = 0.0f;
        float scrollTotal = 0.0f;
        float2 startMousePos = {0.0f, 0.0f};
    } mInstanceTransformationEdit;

    CurrentEdit mCurrentEdit;
    std::vector<SDFEdit> mPerformedSDFEdits;
    std::vector<UndoneSDFEdit> mUndoneSDFEdits;
    bool mLMBDown = false;
    bool mRMBDown = false;
    bool mMMBDown = false;
    bool mEditingKeyDown = false;
    bool mGUIKeyDown = false;
    bool mPreviewEnabled = true;
    bool mAllowEditingOnOtherSurfaces = false;
    bool mAutoBakingEnabled = true;

    uint2 mFrameDim = {0, 0};
    UI2D mUI2D;

    ref<Buffer> mpPickingInfo;         ///< Buffer for reading back picking info from the GPU.
    ref<Buffer> mpPickingInfoReadBack; ///< Staging buffer for reading back picking info from the GPU.
    ref<Fence> mpReadbackFence;        ///< GPU fence for synchronizing picking info readback.
    SDFPickingInfo mPickingInfo;

    SDFEditingData mGPUEditingData;

    ref<Buffer> mpGridInstanceIDsBuffer;
    uint32_t mGridInstanceCount = 0;

    uint32_t mNonBakedPrimitiveCount = 0;
    uint32_t mBakePrimitivesBatchSize = 5; ///< The number of primitives to bake at a time.
    uint32_t mPreservedHistoryCount = 100; ///< Primitives that should not be baked.

    // Undo/Redo
    uint32_t mUndoPressedCount = 0;
    uint32_t mRedoPressedCount = 0;
};
