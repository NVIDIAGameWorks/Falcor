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
#include "SDFEditor.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Scene/SDFs/SDF3DPrimitiveFactory.h"

namespace
{
const std::string kInputColorChannel = "inputColor";
const std::string kInputVBuffer = "vbuffer";
const std::string kInputDepth = "linearZ";
const std::string kOutputChannel = "output";

const Falcor::ChannelList kInputChannels = {
    {kInputVBuffer, "gVBuffer", "Visibility buffer in packed format", false, ResourceFormat::Unknown},
    {kInputDepth, "gLinearZ", "Linear Z and slope", false, ResourceFormat::RG32Float},
    {kInputColorChannel, "gInputColor", "The input image (2D GUI will be drawn on top)", false, ResourceFormat::RGBA32Float},
};

const std::string kGUIPassShaderFilename = "RenderPasses/SDFEditor/GUIPass.ps.slang";

const uint32_t kInvalidPrimitiveID = std::numeric_limits<uint32_t>::max();

const float4 kLineColor = float4(0.585f, 1.0f, 0.0f, 1.0f);
const float4 kMarkerColor = float4(0.9f, 0.9f, 0.9f, 1.0f);
const float4 kSelectionColor = float4(1.0f, 1.0f, 1.0f, 0.75f);
const float4 kCurrentModeBGColor = float4(0.585f, 1.0f, 0.0f, 0.5f);

const float kFadeAwayDuration = 0.25f; // Time (in seconds) when the 2D GUI fades away after the main GUI key has been released.

const float kMarkerSizeFactor = 0.75f;
const float kMarkerSizeFactorSmoothUnion = 0.6f;

const float kScrollTranslationMultiplier = 0.01f;
const float kMaxOpSmoothingRadius = 0.01f;
const float kMinOpSmoothingRadius = 0.0001f;

const float kMinShapeBlobbyness = 0.001f;
const float kMaxShapeBlobbyness = 0.02f;
const float kMinOperationSmoothness = 0.01f;
const float kMaxOperationSmoothness = 0.05f;

const FileDialogFilterVec kSDFFileExtensionFilters = {{"sdf", "SDF Files"}};
const FileDialogFilterVec kSDFGridFileExtensionFilters = {{"sdfg", "SDF Grid Files"}};

bool isOperationSmooth(SDFOperationType operationType)
{
    switch (operationType)
    {
    case SDFOperationType::SmoothUnion:
    case SDFOperationType::SmoothSubtraction:
    case SDFOperationType::SmoothIntersection:
        return true;
    }

    return false;
}

SDF2DShapeType sdf3DTo2DShape(SDF3DShapeType shapeType)
{
    switch (shapeType)
    {
    case SDF3DShapeType::Sphere:
        return SDF2DShapeType::Circle;
    case SDF3DShapeType::Box:
        return SDF2DShapeType::Square;
    default:
        return SDF2DShapeType::Circle;
    }
}
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, SDFEditor>();
    ScriptBindings::registerBinding(SDFEditor::registerBindings);
}

void SDFEditor::registerBindings(pybind11::module& m)
{
    // None at the moment.
}

SDFEditor::SDFEditor(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    mpFbo = Fbo::create(mpDevice);

    mpPickingInfo = mpDevice->createStructuredBuffer(
        sizeof(SDFPickingInfo), 1, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal
    );
    mpPickingInfoReadBack = mpDevice->createStructuredBuffer(sizeof(SDFPickingInfo), 1, ResourceBindFlags::None, MemoryType::ReadBack);
    mpReadbackFence = mpDevice->createFence();

    mUI2D.pMarker2DSet = std::make_unique<Marker2DSet>(mpDevice, 100);
    mUI2D.pSelectionWheel = std::make_unique<SelectionWheel>(*mUI2D.pMarker2DSet);

    mpSDFEditingDataBuffer = mpDevice->createStructuredBuffer(sizeof(SDFEditingData), 1);

    mUI2D.symmetryPlane.normal = float3(1.0f, 0.0f, 0.0f);
    mUI2D.symmetryPlane.rightVector = float3(0.0f, 0.0f, -1.0f);
    mUI2D.symmetryPlane.color = float4(1.0f, 0.75f, 0.8f, 0.5f);
}

void SDFEditor::bindShaderData(const ShaderVar& var, const ref<Texture>& pInputColor, const ref<Texture>& pVBuffer)
{
    mGPUEditingData.editing = mEditingKeyDown;
    mGPUEditingData.previewEnabled = mPreviewEnabled;
    mGPUEditingData.instanceID = mCurrentEdit.instanceID;
    mGPUEditingData.scalingAxis = uint32_t(
        mPrimitiveTransformationEdit.state != TransformationState::Scaling ? SDFEditorAxis::Count : mPrimitiveTransformationEdit.axis
    );
    mGPUEditingData.primitive = mCurrentEdit.primitive;
    mGPUEditingData.primitiveBB = SDF3DPrimitiveFactory::computeAABB(mCurrentEdit.primitive);
    mpSDFEditingDataBuffer->setBlob(&mGPUEditingData, 0, sizeof(SDFEditingData));

    if (!mpGridInstanceIDsBuffer || mGridInstanceCount < mpScene->getSDFGridCount())
    {
        mGridInstanceCount = mpScene->getSDFGridCount(); // This is safe because the SDF Editor only supports SDFGrid type of SBS for now.
        std::vector<uint32_t> instanceIDs = mpScene->getGeometryInstanceIDsByType(Scene::GeometryType::SDFGrid);
        mpGridInstanceIDsBuffer = mpDevice->createStructuredBuffer(
            sizeof(uint32_t),
            mGridInstanceCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            instanceIDs.data(),
            false
        );
    }

    auto rootVar = mpGUIPass->getRootVar();

    rootVar["gInputColor"] = pInputColor;
    rootVar["gLinearZ"] = mpEditingLinearZBuffer;
    rootVar["gVBuffer"] = pVBuffer;
    mpScene->bindShaderData(rootVar["gScene"]);

    auto guiPassVar = rootVar["gGUIPass"];
    guiPassVar["resolution"] = mFrameDim;
    guiPassVar["mousePos"] = uint2(mUI2D.currentMousePosition);
    guiPassVar["pickingData"] = mpPickingInfo;
    guiPassVar["editingPrimitiveData"].setBuffer(mpSDFEditingDataBuffer);
    guiPassVar["gridInstanceIDs"] = mpGridInstanceIDsBuffer;
    guiPassVar["bbRenderSettings"].setBlob(mUI2D.bbRenderSettings);
    guiPassVar["gridInstanceCount"] = mGridInstanceCount;
    guiPassVar["ui2DActive"] = uint(isMainGUIKeyDown());
    guiPassVar["gridPlane"].setBlob(mUI2D.gridPlane);
    guiPassVar["symmetryPlane"].setBlob(mUI2D.symmetryPlane);
    mUI2D.pMarker2DSet->bindShaderData(guiPassVar["markerSet"]);
}

void SDFEditor::fetchPreviousVBufferAndZBuffer(RenderContext* pRenderContext, ref<Texture>& pVBuffer, ref<Texture>& pDepth)
{
    if (!mpEditingVBuffer || mpEditingVBuffer->getWidth() != pVBuffer->getWidth() || mpEditingVBuffer->getHeight() != pVBuffer->getHeight())
    {
        mpEditingVBuffer = mpDevice->createTexture2D(pVBuffer->getWidth(), pVBuffer->getHeight(), pVBuffer->getFormat(), 1, 1);
    }

    if (!mpEditingLinearZBuffer || mpEditingLinearZBuffer->getWidth() != pDepth->getWidth() ||
        mpEditingLinearZBuffer->getHeight() != pDepth->getHeight())
    {
        mpEditingLinearZBuffer = mpDevice->createTexture2D(
            pDepth->getWidth(),
            pDepth->getHeight(),
            ResourceFormat::RG32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
    }

    pRenderContext->copySubresourceRegion(mpEditingVBuffer.get(), 0, pVBuffer.get(), pVBuffer->getSubresourceIndex(0, 0));
    pRenderContext->copySubresourceRegion(mpEditingLinearZBuffer.get(), 0, pDepth.get(), pDepth->getSubresourceIndex(0, 0));
}

Properties SDFEditor::getProperties() const
{
    return {};
}

void SDFEditor::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;

    if (!mpScene)
        return;

    mpCamera = mpScene->getCamera();
    mpGUIPass = FullScreenPass::create(
        mpDevice, ProgramDesc().addShaderLibrary(kGUIPassShaderFilename).psEntry("psMain"), mpScene->getSceneDefines()
    );

    // Initialize editing primitive.
    {
        const SDF3DShapeType kDefaultShapeType = SDF3DShapeType::Sphere;
        const float3 kDefaultShapeData = float3(0.01f);
        const float kDefaultShapeBlobbing = 0.0f;
        const float kDefaultOperationSmoothing = 0.0f;
        const SDFOperationType kDefaultOperationType = SDFOperationType::Union;
        const Transform kDefaultTransform = Transform();

        mUI2D.currentEditingShape = kDefaultShapeType;
        mUI2D.currentEditingOperator = kDefaultOperationType;

        if (mpScene->getSDFGridCount() > 0)
        {
            std::vector<uint32_t> instanceIDs = mpScene->getGeometryInstanceIDsByType(Scene::GeometryType::SDFGrid);
            if (instanceIDs.empty())
                FALCOR_THROW("Scene missing SDFGrid object!");

            mCurrentEdit.instanceID = instanceIDs[0];
            GeometryInstanceData instance = mpScene->getGeometryInstance(mCurrentEdit.instanceID);
            SdfGridID sdfGridID = mpScene->findSDFGridIDFromGeometryInstanceID(mCurrentEdit.instanceID);
            FALCOR_ASSERT(sdfGridID != SdfGridID::Invalid());
            mCurrentEdit.gridID = sdfGridID;
            mCurrentEdit.pSDFGrid = mpScene->getSDFGrid(sdfGridID);
            mCurrentEdit.primitive = SDF3DPrimitiveFactory::initCommon(
                kDefaultShapeType,
                kDefaultShapeData,
                kDefaultShapeBlobbing,
                kDefaultOperationSmoothing,
                kDefaultOperationType,
                kDefaultTransform
            );

            const AnimationController* pAnimationController = mpScene->getAnimationController();
            const float4x4& transform = pAnimationController->getGlobalMatrices()[instance.globalMatrixID];

            // Update GUI variables
            mUI2D.bbRenderSettings.selectedInstanceID = mCurrentEdit.instanceID;
            mUI2D.gridPlane.position = transform.getCol(3).xyz();
            mUI2D.previousGridPlane.position = transform.getCol(3).xyz();

            updateSymmetryPrimitive();
        }
    }

    mNonBakedPrimitiveCount = mCurrentEdit.pSDFGrid->getPrimitiveCount();
}

RenderPassReflection SDFEditor::reflect(const CompileData& compileData)
{
    // Define the required resources here
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    reflector.addOutput(kOutputChannel, "Input image with 2D GUI drawn on top");
    return reflector;
}

bool SDFEditor::isMainGUIKeyDown() const
{
    bool notTransformingPrimitive = mPrimitiveTransformationEdit.state == TransformationState::None;
    return mGUIKeyDown && notTransformingPrimitive && !gridPlaneManipulated() && !symmetryPlaneManipulated() &&
           !mUI2D.keyboardButtonsPressed.undo && !mUI2D.keyboardButtonsPressed.redo;
}

void SDFEditor::updateEditShapeType()
{
    mCurrentEdit.primitive.shapeType = mUI2D.currentEditingShape;
    mCurrentEdit.primitive.shapeBlobbing = mUI2D.currentBlobbing;

    if (mUI2D.symmetryPlane.active)
    {
        updateSymmetryPrimitive();
    }
}

void SDFEditor::updateEditOperationType()
{
    mCurrentEdit.primitive.operationType = mUI2D.currentEditingOperator;
    mCurrentEdit.primitive.operationSmoothing =
        isOperationSmooth(mUI2D.currentEditingOperator) ? 0.5f * (kMaxOperationSmoothness + kMinOperationSmoothness) : 0.0f;

    if (mUI2D.symmetryPlane.active)
    {
        updateSymmetryPrimitive();
    }
}

void SDFEditor::updateSymmetryPrimitive()
{
    const AnimationController* pAnimationController = mpScene->getAnimationController();
    const GeometryInstanceData& instance = mpScene->getGeometryInstance(mCurrentEdit.instanceID);
    float4x4 invInstanceTransform = transpose(pAnimationController->getInvTransposeGlobalMatrices()[instance.globalMatrixID]);

    const float3& instanceLocalPrimitivePos = mCurrentEdit.primitive.translation;
    float3 instanceLocalPlanePosition = transformPoint(invInstanceTransform, mUI2D.symmetryPlane.position);
    float3 instanceLocalPlaneNormal = normalize(transformVector(invInstanceTransform, mUI2D.symmetryPlane.normal));
    float3 projInstanceLocalPrimitivePos =
        (instanceLocalPrimitivePos - dot(instanceLocalPrimitivePos, instanceLocalPlaneNormal) * instanceLocalPlaneNormal);
    float3 projInstanceLocalPlanePos = dot(instanceLocalPlanePosition, instanceLocalPlaneNormal) * instanceLocalPlaneNormal;

    float3 reflectedInstanceLocalPos;
    if (length(projInstanceLocalPrimitivePos) > 0.0f)
    {
        reflectedInstanceLocalPos =
            math::reflect(-(instanceLocalPrimitivePos - projInstanceLocalPlanePos), normalize(projInstanceLocalPrimitivePos));
    }
    else
    {
        reflectedInstanceLocalPos = length(instanceLocalPrimitivePos - projInstanceLocalPlanePos) * -instanceLocalPlaneNormal;
    }

    reflectedInstanceLocalPos += projInstanceLocalPlanePos;

    float3x3 symmetricPrimitiveTransform = float3x3(inverse(mCurrentEdit.primitive.invRotationScale));
    symmetricPrimitiveTransform = math::rotate(float4x4(symmetricPrimitiveTransform), float(M_PI), instanceLocalPlaneNormal);

    mCurrentEdit.symmetryPrimitive = mCurrentEdit.primitive;
    mCurrentEdit.symmetryPrimitive.translation = reflectedInstanceLocalPos;
    mCurrentEdit.symmetryPrimitive.invRotationScale = inverse(float3x3(symmetricPrimitiveTransform));
}

void SDFEditor::setupPrimitiveAndOperation(
    const float2& center,
    const float markerSize,
    SDF3DShapeType editingPrimitive,
    SDFOperationType editingOperator,
    const float4& color,
    const float alpha
)
{
    const float4 dimmedColor = color * float4(0.6f, 0.6f, 0.6f, 1.0f);
    float f = markerSize * 0.25f;
    const float delta = (editingOperator == SDFOperationType::SmoothUnion) ? f * kMarkerSizeFactor : 0.0f;
    f += delta;

    float2 dir = float2(f, f);
    float2 pos1 = center - dir;
    float2 pos2 = center + dir;
    float4 fade = float4(1.0f, 1.0f, 1.0f, alpha);
    mUI2D.pMarker2DSet->addMarkerOpMarker(
        editingOperator,
        sdf3DTo2DShape(editingPrimitive),
        pos1,
        markerSize,
        sdf3DTo2DShape(editingPrimitive),
        pos2,
        markerSize,
        color * fade,
        dimmedColor * fade
    );
}

void SDFEditor::setupCurrentModes2D()
{
    const float minSide = float(std::min(mFrameDim.x, mFrameDim.y));
    const float side = minSide / 6.0f;
    const float markerSize = side * 0.5f / kMarkerSizeFactor;
    const float roundedRadius = markerSize * 0.15f;
    const float cornerOffset = 10.0f;
    const float2 center = float2(side * 0.5f, mFrameDim.y - side * 0.5f) + float2(cornerOffset, -cornerOffset);

    // Setup a rounded box as a background where the selected shape and operation will be displayed.
    mUI2D.pMarker2DSet->addRoundedBox(center, float2(side, side) * 0.5f, roundedRadius, 0.0f, kCurrentModeBGColor); // Add the background
                                                                                                                    // box, in which we will
                                                                                                                    // draw markers.

    // Setup the selected shape and operation markers.
    float f = mUI2D.currentEditingOperator == SDFOperationType::SmoothUnion ? kMarkerSizeFactorSmoothUnion : kMarkerSizeFactor;
    setupPrimitiveAndOperation(center, markerSize * f, mUI2D.currentEditingShape, mUI2D.currentEditingOperator, kMarkerColor);
}

void SDFEditor::manipulateGridPlane(
    SDFGridPlane& gridPlane,
    SDFGridPlane& previousGridPlane,
    bool isTranslationKeyDown,
    bool isConstrainedManipulationKeyDown
)
{
    float2 diffPrev = mUI2D.currentMousePosition - mUI2D.prevMousePosition;
    float2 diffStart = mUI2D.currentMousePosition - mUI2D.startMousePosition;
    float3 up = mpCamera->getUpVector();
    float3 view = normalize(mpCamera->getTarget() - mpCamera->getPosition());
    float3 right = cross(view, up);

    if (!isTranslationKeyDown) // The user wants rotation of the grid plane.
    {
        if (!isConstrainedManipulationKeyDown) // Rotate plane arbitrarily along right and up vectors.
        {
            // Rotate plane around the right vector.
            rotateGridPlane(
                diffPrev.y, right, previousGridPlane.normal, previousGridPlane.rightVector, gridPlane.normal, gridPlane.rightVector
            );
            // Rotate plane around the up vector.
            rotateGridPlane(diffPrev.x, up, gridPlane.normal, gridPlane.rightVector, gridPlane.normal, gridPlane.rightVector);
            previousGridPlane = gridPlane;
        }
        else // Constraind rotation to the axis with most movement since shift was pressed.
        {
            if (std::abs(diffStart.y) > std::abs(diffStart.x)) // Rotate plane only around the right vector.
            {
                rotateGridPlane(
                    diffStart.y,
                    right,
                    previousGridPlane.normal,
                    previousGridPlane.rightVector,
                    gridPlane.normal,
                    gridPlane.rightVector,
                    false
                );
            }
            else // Rotate plane only around the up vector.
            {
                rotateGridPlane(
                    diffStart.x, up, previousGridPlane.normal, previousGridPlane.rightVector, gridPlane.normal, gridPlane.rightVector, false
                );
            }
        }
    }
    else // The user wants translation of the grid plane.
    {
        if (!isConstrainedManipulationKeyDown)
        {
            translateGridPlane(diffPrev.x, right, previousGridPlane.position, gridPlane.position);
            translateGridPlane(-diffPrev.y, up, gridPlane.position, gridPlane.position);
            previousGridPlane = gridPlane;
        }
        else
        {
            if (std::abs(diffStart.y) > std::abs(diffStart.x)) // Translate plane only along the right vector.
            {
                translateGridPlane(-diffStart.y, up, previousGridPlane.position, gridPlane.position);
            }
            else // Translate plane only along the up vector.
            {
                translateGridPlane(diffStart.x, right, previousGridPlane.position, gridPlane.position);
            }
        }
    }
}

void SDFEditor::rotateGridPlane(
    const float mouseDiff,
    const float3& rotationVector,
    const float3& inNormal,
    const float3& inRightVector,
    float3& outNormal,
    float3& outRightVector,
    const bool fromPreviousMouse
)
{
    const float diagonal = length(float2(mFrameDim));
    const float maxAngle = float(M_PI) * 0.05f;
    float angle;

    if (fromPreviousMouse)
    {
        const float speedFactor = 2.0f * float(M_PI) * 0.075f / diagonal;
        angle = mouseDiff * std::abs(mouseDiff) * speedFactor;
        angle = std::clamp(angle, -maxAngle, maxAngle);
    }
    else // From the start position -- which is better when doing constrained rotation (around a single axis).
    {
        const float speedFactor = 2.0f * float(M_PI) * 0.5f / diagonal;
        angle = mouseDiff * speedFactor;
    }
    float4x4 rotationMatrix = math::rotate(float4x4::identity(), angle, rotationVector);

    outNormal = normalize(transformVector(rotationMatrix, inNormal));
    outRightVector = normalize(transformVector(rotationMatrix, inRightVector));
}

void SDFEditor::translateGridPlane(const float mouseDiff, const float3& translationVector, const float3& inPosition, float3& outPosition)
{
    const float diagonal = length(float2(mFrameDim));
    const float speedFactor = 0.5f / diagonal;
    float translation = mouseDiff * std::abs(mouseDiff) * speedFactor; // Could possibly be improved so that speed depends on scene or size
                                                                       // of the grid.
    outPosition = outPosition + translation * translationVector;
}

bool SDFEditor::gridPlaneManipulated() const
{
    return mUI2D.gridPlane.active && mRMBDown;
}

bool SDFEditor::symmetryPlaneManipulated() const
{
    return mUI2D.symmetryPlane.active && mMMBDown;
}

void SDFEditor::setup2DGUI()
{
    const float2 center = float2(mFrameDim) * 0.5f;
    const float minSide = float(std::min(mFrameDim.x, mFrameDim.y));
    const float radius = 0.5f * minSide;
    const float markerSize = radius * 0.2f;
    const float roundedRadius = markerSize * 0.1f;

    mUI2D.pMarker2DSet->clear();

    // Draw editing menu wheel.
    if (isMainGUIKeyDown() || mUI2D.fadeAwayGUI)
    {
        float alpha = 1.0f;
        if (!isMainGUIKeyDown())
        {
            CpuTimer::TimePoint t = mUI2D.timer.getCurrentTimePoint();
            float deltaTime = float(CpuTimer::calcDuration(mUI2D.timeOfReleaseMainGUIKey, t)) * 0.001f; // In seconds.
            if (deltaTime >= kFadeAwayDuration)
            {
                mUI2D.fadeAwayGUI = false;
                alpha = 0.0f;
            }
            else
            {
                alpha = 1.0f - deltaTime / kFadeAwayDuration; // In [0,1].
            }
        }

        float4 multColor = float4(1.0f, 1.0f, 1.0f, alpha);
        float4 color = kMarkerColor * multColor;

        SelectionWheel::Desc swDesc;
        swDesc.position = center;
        swDesc.minRadius = 0.3f * radius;
        swDesc.maxRadius = 0.7f * radius;
        swDesc.baseColor = float4(0.13f, 0.13f, 0.1523f, 0.8f * alpha);
        swDesc.highlightColor = float4(0.4648f, 0.7226f, 0.0f, 0.8f * alpha);
        swDesc.sectorGroups = {2, 4};
        swDesc.lineColor = kLineColor * multColor;
        swDesc.borderWidth = 10.0f;
        mUI2D.pSelectionWheel->update(mUI2D.currentMousePosition, swDesc);
        mUI2D.pMarker2DSet->addSimpleMarker(
            SDF2DShapeType::Square, markerSize, mUI2D.pSelectionWheel->getCenterPositionOfSector(0, 0), 0.0f, color
        );
        mUI2D.pMarker2DSet->addSimpleMarker(
            SDF2DShapeType::Circle, markerSize * 0.5f, mUI2D.pSelectionWheel->getCenterPositionOfSector(0, 1), 0.0f, color
        );
        setupPrimitiveAndOperation(
            mUI2D.pSelectionWheel->getCenterPositionOfSector(1, 0),
            markerSize * kMarkerSizeFactor,
            mUI2D.currentEditingShape,
            SDFOperationType::SmoothSubtraction,
            kMarkerColor,
            alpha
        );
        setupPrimitiveAndOperation(
            mUI2D.pSelectionWheel->getCenterPositionOfSector(1, 1),
            markerSize * kMarkerSizeFactor,
            mUI2D.currentEditingShape,
            SDFOperationType::Subtraction,
            kMarkerColor,
            alpha
        );
        setupPrimitiveAndOperation(
            mUI2D.pSelectionWheel->getCenterPositionOfSector(1, 2),
            markerSize * kMarkerSizeFactor,
            mUI2D.currentEditingShape,
            SDFOperationType::Union,
            kMarkerColor,
            alpha
        );
        setupPrimitiveAndOperation(
            mUI2D.pSelectionWheel->getCenterPositionOfSector(1, 3),
            markerSize * kMarkerSizeFactorSmoothUnion,
            mUI2D.currentEditingShape,
            SDFOperationType::SmoothUnion,
            kMarkerColor,
            alpha
        );
    }

    auto addMarker = [&](const SDF3DShapeType& type) -> void
    {
        switch (type)
        {
        case SDF3DShapeType::Box:
            mUI2D.pMarker2DSet->addSimpleMarker(SDF2DShapeType::Square, markerSize, center, 0.0f, kSelectionColor);
            break;
        case SDF3DShapeType::Sphere:
            mUI2D.pMarker2DSet->addSimpleMarker(SDF2DShapeType::Circle, markerSize * 0.5f, center, 0.0f, kSelectionColor);
            break;
        default:
            FALCOR_UNREACHABLE();
            break;
        }
    };

    auto addOperation = [&](const SDFOperationType& type) -> void
    {
        switch (type)
        {
        case SDFOperationType::SmoothSubtraction:
        case SDFOperationType::Subtraction:
        case SDFOperationType::Union:
            setupPrimitiveAndOperation(center, markerSize * kMarkerSizeFactor, mUI2D.currentEditingShape, type, kSelectionColor);
            break;
        case SDFOperationType::SmoothUnion:
            setupPrimitiveAndOperation(center, markerSize * kMarkerSizeFactorSmoothUnion, mUI2D.currentEditingShape, type, kSelectionColor);
            break;
        default:
            FALCOR_UNREACHABLE();
            break;
        }
    };

    static constexpr std::array<SDF3DShapeType, 3> kShapeTypes = {SDF3DShapeType::Box, SDF3DShapeType::Sphere};
    static constexpr std::array<SDFOperationType, 4> kOperationTypes = {
        SDFOperationType::SmoothSubtraction, SDFOperationType::Subtraction, SDFOperationType::Union, SDFOperationType::SmoothUnion};

    // Check if the user pressed on the different sectors of the wheel.
    if (isMainGUIKeyDown() && !mUI2D.recordStartingMousePos && any(mUI2D.startMousePosition != mUI2D.currentMousePosition))
    {
        uint32_t sectorIndex = UINT32_MAX;
        if (mUI2D.pSelectionWheel->isMouseOnGroup(mUI2D.currentMousePosition, 0, sectorIndex))
        {
            if (sectorIndex < kShapeTypes.size())
            {
                addMarker(kShapeTypes[sectorIndex]);
                if (mLMBDown)
                {
                    mUI2D.currentEditingShape = kShapeTypes[sectorIndex];
                    updateEditShapeType();
                }
            }
        }

        if (mUI2D.pSelectionWheel->isMouseOnGroup(mUI2D.currentMousePosition, 1, sectorIndex))
        {
            if (sectorIndex < kOperationTypes.size())
            {
                addOperation(kOperationTypes[sectorIndex]);
                if (mLMBDown)
                {
                    mUI2D.currentEditingOperator = kOperationTypes[sectorIndex];
                    updateEditOperationType();
                }
            }
        }
    }

    if (mUI2D.drawCurrentModes)
    {
        setupCurrentModes2D();
    }
}

void SDFEditor::handleActions()
{
    if (!mpScene)
        return;

    const AnimationController* pAnimationController = mpScene->getAnimationController();
    const GeometryInstanceData& instance = mpScene->getGeometryInstance(mCurrentEdit.instanceID);
    const float4x4& instanceTransform = pAnimationController->getGlobalMatrices()[instance.globalMatrixID];
    Ray ray = mpScene->getCamera()->computeRayPinhole(uint2(mUI2D.currentMousePosition), mFrameDim, false);

    // Rescale grid instance if the grid resolution has changed.
    float resolutionScalingFactor = mCurrentEdit.pSDFGrid->getResolutionScalingFactor();
    if (std::fabs(resolutionScalingFactor - 1.0f) > FLT_EPSILON)
    {
        float3 scale;
        quatf rotation;
        float3 translation;
        float3 skew;
        float4 perspective;
        math::decompose(instanceTransform, scale, rotation, translation, skew, perspective);

        Transform finalTranform;
        finalTranform.setTranslation(translation);
        finalTranform.setRotation(rotation);
        finalTranform.setScaling(scale * resolutionScalingFactor);

        mpScene->updateNodeTransform(instance.globalMatrixID, finalTranform.getMatrix());
        mCurrentEdit.pSDFGrid->resetResolutionScalingFactor();
    }

    // Handle transformation of the SDF grid instance.
    if (mInstanceTransformationEdit.state != TransformationState::None)
    {
        // Initiating starting transformation.
        if (mInstanceTransformationEdit.prevState == TransformationState::None)
        {
            mInstanceTransformationEdit.scrollTotal = 0.0f;

            float3 scale;
            quatf rotation;
            float3 translation;
            float3 skew;
            float4 perspective;
            math::decompose(instanceTransform, scale, rotation, translation, skew, perspective);

            mInstanceTransformationEdit.startTransform.setTranslation(translation);
            mInstanceTransformationEdit.startTransform.setRotation(rotation);
            mInstanceTransformationEdit.startTransform.setScaling(scale);

            float3 deltaCenter = translation - ray.origin;
            float3 planeNormal = -normalize(mpScene->getCamera()->getTarget() - mpScene->getCamera()->getPosition());

            float startT = dot(deltaCenter, planeNormal) / dot(ray.dir, planeNormal);
            mInstanceTransformationEdit.startPlanePos = ray.origin + ray.dir * startT;
            mInstanceTransformationEdit.startMousePos = mUI2D.currentMousePosition;

            float3 arbitraryVectorNotOrthToPlane =
                std::abs(planeNormal.z) < FLT_EPSILON ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
            mInstanceTransformationEdit.referencePlaneDir =
                normalize(arbitraryVectorNotOrthToPlane - dot(arbitraryVectorNotOrthToPlane, planeNormal) * planeNormal);
        }

        // Transform the instance if the mouse was moved or scroll was used.
        if ((any(mUI2D.currentMousePosition != mUI2D.prevMousePosition) ||
             mInstanceTransformationEdit.prevScrollTotal != mInstanceTransformationEdit.scrollTotal))
        {
            const float3& startTranslation = mInstanceTransformationEdit.startTransform.getTranslation();
            float3 deltaCenter = startTranslation - ray.origin;
            float3 planeNormal = -normalize(mpScene->getCamera()->getTarget() - mpScene->getCamera()->getPosition());

            float t = dot(deltaCenter, planeNormal) / dot(ray.dir, planeNormal);

            Transform finalTranform = mInstanceTransformationEdit.startTransform;

            // Handle translation
            if (mInstanceTransformationEdit.state == TransformationState::Translating)
            {
                float3 p = ray.origin + ray.dir * (t + mInstanceTransformationEdit.scrollTotal * kScrollTranslationMultiplier);
                finalTranform.setTranslation(p);
            }
            // Handle rotation
            else if (mInstanceTransformationEdit.state == TransformationState::Rotating)
            {
                float3 p = ray.origin + ray.dir * t;

                float3 startPlaneDir = normalize(mInstanceTransformationEdit.startPlanePos - startTranslation);
                float3 currPlaneDir = normalize(p - startTranslation);

                float startAngle = std::atan2(
                    dot(planeNormal, cross(startPlaneDir, mInstanceTransformationEdit.referencePlaneDir)),
                    dot(startPlaneDir, mInstanceTransformationEdit.referencePlaneDir)
                );
                float currentAngle = std::atan2(
                    dot(planeNormal, cross(currPlaneDir, mInstanceTransformationEdit.referencePlaneDir)),
                    dot(currPlaneDir, mInstanceTransformationEdit.referencePlaneDir)
                );
                float deltaAngle = startAngle - currentAngle;
                float3 localPlaneNormal =
                    normalize(transformVector(inverse(mInstanceTransformationEdit.startTransform.getMatrix()), planeNormal));

                finalTranform.setRotation(
                    mul(mInstanceTransformationEdit.startTransform.getRotation(), math::quatFromAngleAxis(deltaAngle, localPlaneNormal))
                );
            }
            // Handle scaling
            else if (mInstanceTransformationEdit.state == TransformationState::Scaling)
            {
                float deltaX = mUI2D.currentMousePosition.x - mInstanceTransformationEdit.startMousePos.x;
                const float scaleDelta = 1.05f;
                float scale = deltaX < 0.0f ? 1.0f / scaleDelta : scaleDelta;
                scale = std::pow(scale, 100.0f * std::abs(deltaX) / mFrameDim.x);
                finalTranform.setScaling(mInstanceTransformationEdit.startTransform.getScaling() * scale);
            }
            mpScene->updateNodeTransform(instance.globalMatrixID, finalTranform.getMatrix());
        }
    }
    // Handle transformation of the current SDF primitive.
    else if (mPrimitiveTransformationEdit.state != TransformationState::None)
    {
        // Initiate starting transformation for the primitive.
        if (mPrimitiveTransformationEdit.prevState == TransformationState::None)
        {
            mPrimitiveTransformationEdit.startPrimitive = mCurrentEdit.primitive;

            float3 instanceScale;
            quatf instanceRotation;
            float3 instanceTranslation;
            float3 dummySkew;
            float4 dummyPerspective;
            math::decompose(instanceTransform, instanceScale, instanceRotation, instanceTranslation, dummySkew, dummyPerspective);

            mPrimitiveTransformationEdit.startInstanceTransform.setTranslation(instanceTranslation);
            mPrimitiveTransformationEdit.startInstanceTransform.setRotation(instanceRotation);
            mPrimitiveTransformationEdit.startInstanceTransform.setScaling(instanceScale);

            float3 primitiveScale;
            quatf primitiveRotation;
            float3 dummyTranslation;
            float4x4 primitiveTransform = inverse(transpose(mCurrentEdit.primitive.invRotationScale));
            math::decompose(primitiveTransform, primitiveScale, primitiveRotation, dummyTranslation, dummySkew, dummyPerspective);

            mPrimitiveTransformationEdit.startPrimitiveTransform.setTranslation(mCurrentEdit.primitive.translation);
            mPrimitiveTransformationEdit.startPrimitiveTransform.setRotation(primitiveRotation);
            mPrimitiveTransformationEdit.startPrimitiveTransform.setScaling(primitiveScale);

            float3 planeOrigin =
                transformPoint(mPrimitiveTransformationEdit.startInstanceTransform.getMatrix(), mCurrentEdit.primitive.translation);
            float3 deltaCenter = planeOrigin - ray.origin;
            float3 planeNormal = -normalize(mpScene->getCamera()->getTarget() - mpScene->getCamera()->getPosition());

            float startT = dot(deltaCenter, planeNormal) / dot(ray.dir, planeNormal);
            mPrimitiveTransformationEdit.startPlanePos = ray.origin + ray.dir * startT;
            mPrimitiveTransformationEdit.startMousePos = mUI2D.currentMousePosition;

            float3 arbitraryVectorNotOrthToPlane =
                std::abs(planeNormal.z) < FLT_EPSILON ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
            mPrimitiveTransformationEdit.referencePlaneDir =
                normalize(arbitraryVectorNotOrthToPlane - dot(arbitraryVectorNotOrthToPlane, planeNormal) * planeNormal);
        }
        // Update starting position if the state or axis has changed.
        else if (mPrimitiveTransformationEdit.state != mPrimitiveTransformationEdit.prevState || mPrimitiveTransformationEdit.axis != mPrimitiveTransformationEdit.prevAxis)
        {
            mCurrentEdit.primitive = mPrimitiveTransformationEdit.startPrimitive;

            if (mUI2D.symmetryPlane.active)
            {
                updateSymmetryPrimitive();
            }

            float3 planeOrigin =
                transformPoint(mPrimitiveTransformationEdit.startInstanceTransform.getMatrix(), mCurrentEdit.primitive.translation);
            float3 deltaCenter = planeOrigin - ray.origin;
            float3 planeNormal = -normalize(mpScene->getCamera()->getTarget() - mpScene->getCamera()->getPosition());

            float startT = dot(deltaCenter, planeNormal) / dot(ray.dir, planeNormal);
            mPrimitiveTransformationEdit.startPlanePos = ray.origin + ray.dir * startT;
            mPrimitiveTransformationEdit.startMousePos = mUI2D.currentMousePosition;

            float3 arbitraryVectorNotOrthToPlane =
                std::abs(planeNormal.z) < FLT_EPSILON ? float3(0.0f, 0.0f, 1.0f) : float3(1.0f, 0.0f, 0.0f);
            mPrimitiveTransformationEdit.referencePlaneDir =
                normalize(arbitraryVectorNotOrthToPlane - dot(arbitraryVectorNotOrthToPlane, planeNormal) * planeNormal);
        }

        // Transform the primitive if the mouse has moved.
        if (any(mUI2D.currentMousePosition != mUI2D.prevMousePosition))
        {
            float3 planeOrigin = transformPoint(
                mPrimitiveTransformationEdit.startInstanceTransform.getMatrix(),
                mPrimitiveTransformationEdit.startPrimitiveTransform.getTranslation()
            );

            float3 deltaCenter = planeOrigin - ray.origin;
            float3 planeNormal = -normalize(mpScene->getCamera()->getTarget() - mpScene->getCamera()->getPosition());

            float t = dot(deltaCenter, planeNormal) / dot(ray.dir, planeNormal);

            Transform finalPrimitiveTransform = mPrimitiveTransformationEdit.startPrimitiveTransform;

            if (mPrimitiveTransformationEdit.state == TransformationState::Rotating)
            {
                float3 p = ray.origin + ray.dir * t;

                float3 startPlaneDir = mPrimitiveTransformationEdit.startPlanePos - planeOrigin;
                float3 currPlaneDir = p - planeOrigin;

                if (!all(startPlaneDir == float3(0.0f)) && !all(currPlaneDir == float3(0.0f)))
                {
                    startPlaneDir = normalize(startPlaneDir);
                    currPlaneDir = normalize(currPlaneDir);

                    float startAngle = std::atan2(
                        dot(planeNormal, cross(startPlaneDir, mPrimitiveTransformationEdit.referencePlaneDir)),
                        dot(startPlaneDir, mPrimitiveTransformationEdit.referencePlaneDir)
                    );
                    float currentAngle = std::atan2(
                        dot(planeNormal, cross(currPlaneDir, mPrimitiveTransformationEdit.referencePlaneDir)),
                        dot(currPlaneDir, mPrimitiveTransformationEdit.referencePlaneDir)
                    );
                    float deltaAngle = currentAngle - startAngle;
                    float3 localPlaneNormal =
                        normalize(transformVector(inverse(mPrimitiveTransformationEdit.startInstanceTransform.getMatrix()), planeNormal));

                    finalPrimitiveTransform.setRotation(
                        mul(mPrimitiveTransformationEdit.startPrimitiveTransform.getRotation(),
                            math::quatFromAngleAxis(-deltaAngle, localPlaneNormal))
                    );

                    mCurrentEdit.primitive.invRotationScale = transpose(inverse(float3x3(finalPrimitiveTransform.getMatrix())));
                }
            }
            else if (mPrimitiveTransformationEdit.state == TransformationState::Scaling)
            {
                const SDF3DPrimitive& startPrimitive = mPrimitiveTransformationEdit.startPrimitive;
                float deltaX = mUI2D.currentMousePosition.x - mPrimitiveTransformationEdit.startMousePos.x;
                const float scaleDelta = 1.05f;
                float scale = deltaX < 0.0f ? 1.0f / scaleDelta : scaleDelta;
                scale = std::pow(scale, 100.0f * std::abs(deltaX) / mFrameDim.x);
                if (mPrimitiveTransformationEdit.axis == SDFEditorAxis::All)
                {
                    float3x3 scaleMatrix = matrixFromDiagonal(float3(scale));
                    mCurrentEdit.primitive.invRotationScale =
                        transpose(inverse(mul(float3x3(mPrimitiveTransformationEdit.startPrimitiveTransform.getMatrix()), scaleMatrix)));
                }
                else if (mPrimitiveTransformationEdit.axis == SDFEditorAxis::OpSmoothing)
                {
                    float finalScaling =
                        std::clamp(startPrimitive.operationSmoothing * scale, kMinOpSmoothingRadius, kMaxOpSmoothingRadius);
                    mCurrentEdit.primitive.operationSmoothing = finalScaling;
                }
                else
                {
                    uint32_t axis = uint32_t(mPrimitiveTransformationEdit.axis);
                    float3x3 scaleMtx = float3x3::identity();
                    scaleMtx[axis][axis] = scale;
                    mCurrentEdit.primitive.invRotationScale =
                        transpose(inverse(mul(float3x3(mPrimitiveTransformationEdit.startPrimitiveTransform.getMatrix()), scaleMtx)));
                }
            }

            if (mUI2D.symmetryPlane.active)
            {
                updateSymmetryPrimitive();
            }
        }
    }
}

void SDFEditor::handleToggleSymmetryPlane()
{
    mUI2D.symmetryPlane.active = !mUI2D.symmetryPlane.active;

    if (mUI2D.symmetryPlane.active)
    {
        updateSymmetryPrimitive();
    }

    if (mEditingKeyDown)
    {
        if (mUI2D.symmetryPlane.active)
        {
            uint32_t basePrimitiveID;
            basePrimitiveID = mCurrentEdit.pSDFGrid->addPrimitives({mCurrentEdit.symmetryPrimitive});
            mCurrentEdit.symmetryPrimitiveID = basePrimitiveID;
        }
        else if (mCurrentEdit.primitiveID != kInvalidPrimitiveID)
        {
            mCurrentEdit.pSDFGrid->removePrimitives({mCurrentEdit.symmetryPrimitiveID});
            mCurrentEdit.symmetryPrimitiveID = kInvalidPrimitiveID;
        }
    }
}

uint32_t SDFEditor::calcPrimitivesAffectedCount(uint32_t keyPressedCount)
{
    return std::min(std::max(keyPressedCount / 7u, 1u), 10u);
}

void SDFEditor::handleUndo()
{
    uint32_t primitivesAffectedCount = calcPrimitivesAffectedCount(mUndoPressedCount++);

    // Undo - Go though the performed SDF edits, removed them and put them in the mUndoneSDFEdits list.
    std::unordered_map<SdfGridID, std::vector<uint32_t>> primitiveIDsToRemovePerSDF;

    if (mNonBakedPrimitiveCount > 0)
    {
        uint32_t editsToUndoCount = std::min(primitivesAffectedCount, uint32_t(mPerformedSDFEdits.size()));
        editsToUndoCount = std::min(editsToUndoCount, mNonBakedPrimitiveCount);
        for (uint32_t i = 0; i < editsToUndoCount; i++)
        {
            SDFEdit& sdfEdit = mPerformedSDFEdits.back();
            primitiveIDsToRemovePerSDF[sdfEdit.gridID].push_back(sdfEdit.primitiveID);
            mPerformedSDFEdits.pop_back();
        }

        for (auto& pair : primitiveIDsToRemovePerSDF)
        {
            SdfGridID gridID = pair.first;
            const std::vector<uint32_t>& primitiveIDsToRemove = pair.second;

            const ref<SDFGrid>& pSDFGrid = mpScene->getSDFGrid(gridID);

            UndoneSDFEdit undoEdit;
            undoEdit.gridID = gridID;
            for (uint32_t primitiveToRemove : primitiveIDsToRemove)
            {
                undoEdit.primitive = pSDFGrid->getPrimitive(primitiveToRemove);
                mUndoneSDFEdits.push_back(undoEdit);
                mNonBakedPrimitiveCount--;
            }
            pSDFGrid->removePrimitives(primitiveIDsToRemove);
        }
    }
}

void SDFEditor::handleRedo()
{
    uint32_t primitivesAffectedCount = calcPrimitivesAffectedCount(mRedoPressedCount++);

    // Redo - Go through the undone SDF edits and put them back into the mPerformedSDFEdits list.
    std::unordered_map<SdfGridID, std::vector<SDF3DPrimitive>> primitivesToAddPerSDF;

    uint32_t editsToRedoCount = std::min(primitivesAffectedCount, uint32_t(mUndoneSDFEdits.size()));
    for (uint32_t i = 0; i < editsToRedoCount; i++)
    {
        UndoneSDFEdit undoneEdit = mUndoneSDFEdits.back();
        primitivesToAddPerSDF[undoneEdit.gridID].push_back(undoneEdit.primitive);
        mUndoneSDFEdits.pop_back();
    }

    for (auto pair : primitivesToAddPerSDF)
    {
        SdfGridID gridID = pair.first;
        const std::vector<SDF3DPrimitive>& primitivesToAdd = pair.second;

        uint32_t basePrimitiveID;
        basePrimitiveID = mpScene->getSDFGrid(gridID)->addPrimitives(primitivesToAdd);
        mNonBakedPrimitiveCount++;

        SDFEdit redoEdit;
        redoEdit.gridID = gridID;

        for (uint32_t id = basePrimitiveID; id < basePrimitiveID + primitivesToAdd.size(); id++)
        {
            redoEdit.primitiveID = id;
            mPerformedSDFEdits.push_back(redoEdit);
        }
    }
}

void SDFEditor::bakePrimitives()
{
    // Bake primitives that exceed the preserved history count in batches.
    if (mNonBakedPrimitiveCount > mBakePrimitivesBatchSize + mPreservedHistoryCount)
    {
        uint32_t batchCount = (mNonBakedPrimitiveCount - mPreservedHistoryCount) / mBakePrimitivesBatchSize;
        uint32_t bakePrimitivesCount = mBakePrimitivesBatchSize * batchCount;
        mCurrentEdit.pSDFGrid->bakePrimitives(bakePrimitivesCount);
        mNonBakedPrimitiveCount -= bakePrimitivesCount;
    }
}

void SDFEditor::addEditPrimitive(bool addToCurrentEdit, bool addToHistory)
{
    uint32_t basePrimitiveID = kInvalidPrimitiveID;
    if (mUI2D.symmetryPlane.active)
    {
        updateSymmetryPrimitive();
        basePrimitiveID = mCurrentEdit.pSDFGrid->addPrimitives({mCurrentEdit.primitive, mCurrentEdit.symmetryPrimitive});

        if (addToCurrentEdit)
        {
            mCurrentEdit.primitiveID = basePrimitiveID;
            mCurrentEdit.symmetryPrimitiveID = basePrimitiveID + 1;
        }

        if (addToHistory)
        {
            SDFEdit edit;
            edit.gridID = mCurrentEdit.gridID;
            edit.primitiveID = basePrimitiveID;

            // Main edit.
            mPerformedSDFEdits.push_back(edit);

            // Symmetry edit.
            edit.primitiveID++;
            mPerformedSDFEdits.push_back(edit);
        }

        mNonBakedPrimitiveCount += 2;
    }
    else
    {
        basePrimitiveID = mCurrentEdit.pSDFGrid->addPrimitives({mCurrentEdit.primitive});

        if (addToCurrentEdit)
        {
            mCurrentEdit.primitiveID = basePrimitiveID;
        }

        if (addToHistory)
        {
            SDFEdit edit;
            edit.gridID = mCurrentEdit.gridID;
            edit.primitiveID = basePrimitiveID;
            mPerformedSDFEdits.push_back(edit);
        }

        mNonBakedPrimitiveCount++;
    }
}

void SDFEditor::removeEditPrimitives()
{
    if (mUI2D.symmetryPlane.active)
    {
        if (mCurrentEdit.primitiveID != kInvalidPrimitiveID && mCurrentEdit.symmetryPrimitiveID != kInvalidPrimitiveID)
        {
            mCurrentEdit.pSDFGrid->removePrimitives({mCurrentEdit.primitiveID, mCurrentEdit.symmetryPrimitiveID});
            mCurrentEdit.primitiveID = kInvalidPrimitiveID;
            mCurrentEdit.symmetryPrimitiveID = kInvalidPrimitiveID;
            mNonBakedPrimitiveCount -= 2;
        }
    }
    else
    {
        if (mCurrentEdit.primitiveID != kInvalidPrimitiveID)
        {
            mCurrentEdit.pSDFGrid->removePrimitives({mCurrentEdit.primitiveID});
            mCurrentEdit.primitiveID = kInvalidPrimitiveID;
            mNonBakedPrimitiveCount--;
        }
    }
}

void SDFEditor::updateEditPrimitives()
{
    mCurrentEdit.pSDFGrid->updatePrimitives({{mCurrentEdit.primitiveID, mCurrentEdit.primitive}});
    if (mUI2D.symmetryPlane.active)
    {
        updateSymmetryPrimitive();
        mCurrentEdit.pSDFGrid->updatePrimitives({{mCurrentEdit.symmetryPrimitiveID, mCurrentEdit.symmetryPrimitive}});
    }
}

void SDFEditor::handleToggleEditing()
{
    // Toggle the primitive preview by adding a primitive to the current edit or removing it depending on if the editing mode is active.
    if (mEditingKeyDown)
    {
        if (!handlePicking(mUI2D.currentMousePosition, mCurrentEdit.primitive.translation))
            return;
        addEditPrimitive(true, false);
    }
    else
    {
        removeEditPrimitives();
    }
}

void SDFEditor::handleEditMovement()
{
    // Update the current primitive to the current mouse position projected onto the surface.
    if (!handlePicking(mUI2D.currentMousePosition, mCurrentEdit.primitive.translation))
        return;

    if (mCurrentEdit.primitiveID != kInvalidPrimitiveID)
    {
        updateEditPrimitives();
    }
}

void SDFEditor::handleAddPrimitive()
{
    // Add a primitive on the current mouse position projected onto the surface.
    float3 localPos;
    if (!handlePicking(mUI2D.currentMousePosition, localPos))
        return;

    mCurrentEdit.primitive.translation = localPos;
    updateEditPrimitives();
    addEditPrimitive(false, true);

    if (mAutoBakingEnabled)
    {
        bakePrimitives();
    }
}

bool SDFEditor::handlePicking(const float2& currentMousePos, float3& localPos)
{
    if (!mpScene)
        return false;

    // Create picking ray.
    float3 rayOrigin = mpCamera->getPosition();

    const CameraData& cameraData = mpCamera->getData();
    float2 ndc = float2(-1.0f, 1.0f) + float2(2.0f, -2.0f) * (currentMousePos + float2(0.5f, 0.5f)) / float2(mFrameDim);
    float3 rayDir = normalize(ndc.x * cameraData.cameraU + ndc.y * cameraData.cameraV + cameraData.cameraW);

    float3 iSectPosition;
    if (mUI2D.gridPlane.active) // Grid is on, so we pick on the grid.
    {
        float t = mUI2D.gridPlane.intersect(rayOrigin, rayDir);
        iSectPosition = rayOrigin + rayDir * t;
    }
    else if (mAllowEditingOnOtherSurfaces || mPickingInfo.instanceID == mCurrentEdit.instanceID)
    {
        iSectPosition = rayOrigin + rayDir * mPickingInfo.distance;
    }
    else
    {
        return false;
    }

    const GeometryInstanceData& instance = mpScene->getGeometryInstance(mCurrentEdit.instanceID);
    const AnimationController* pAnimationController = mpScene->getAnimationController();
    const float4x4& invTransposeInstanceTransform = pAnimationController->getInvTransposeGlobalMatrices()[instance.globalMatrixID];
    localPos = transformPoint(transpose(invTransposeInstanceTransform), iSectPosition);
    return true;
}

void SDFEditor::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpScene || !mpGUIPass)
        return;

    auto pOutput = renderData.getTexture(kOutputChannel);
    auto pInputColor = renderData.getTexture(kInputColorChannel);
    FALCOR_ASSERT(pOutput && pInputColor);

    mFrameDim = uint2(pOutput->getWidth(), pOutput->getHeight());
    mpFbo->attachColorTarget(pOutput, 0);

    // Make a copy of the vBuffer and the linear z-buffer before editing.
    ref<Texture> pVBuffer = renderData.getTexture(kInputVBuffer);
    ref<Texture> pDepth = renderData.getTexture(kInputDepth);
    if (!mEditingKeyDown)
    {
        fetchPreviousVBufferAndZBuffer(pRenderContext, pVBuffer, pDepth);
    }
    else // When editing, use the copies instead to not add SDF primitives on the current edits.
    {
        pVBuffer = mpEditingVBuffer;
    }

    // Wait for the picking info from the previous frame.
    mpReadbackFence->wait();
    mPickingInfo = *reinterpret_cast<const SDFPickingInfo*>(mpPickingInfoReadBack->map(Buffer::MapType::Read));
    mpPickingInfoReadBack->unmap();

    setup2DGUI();
    handleActions();

    // Set shader data.
    auto rootVar = mpGUIPass->getRootVar();
    bindShaderData(rootVar, pInputColor, pVBuffer);

    mpGUIPass->execute(pRenderContext, mpFbo);

    // Copy picking info into a staging buffer.
    pRenderContext->copyResource(mpPickingInfoReadBack.get(), mpPickingInfo.get());
    pRenderContext->submit(false);
    pRenderContext->signal(mpReadbackFence.get());

    // Prepare next frame.
    {
        mUI2D.keyboardButtonsPressed.registerCurrentStatesIntoPrevious();

        mUI2D.prevMousePosition = mUI2D.currentMousePosition;
        mUI2D.scrollDelta = 0.0f;
        mInstanceTransformationEdit.prevState = mInstanceTransformationEdit.state;
        mInstanceTransformationEdit.prevScrollTotal = mInstanceTransformationEdit.scrollTotal;
        mPrimitiveTransformationEdit.prevState = mPrimitiveTransformationEdit.state;
        mPrimitiveTransformationEdit.prevAxis = mPrimitiveTransformationEdit.axis;
    }
}

void SDFEditor::renderUI(RenderContext* pRenderContext, Gui::Widgets& widget)
{
    widget.text("Help:");
    widget.tooltip(
        "Abbreviations: Prim=Primitive, Op=Operation, MB = mouse button, LMB = left MB, RMB = right MB, MMB = middle MB\n"
        "\n"
        "To create an SDF from an empty grid, use either the grid plane to place the SDF on, or intersect the grid instance with other "
        "geometry and toggle Editing on Other surfaces by pressing 'C'.\n"
        "\n"
        "* Bring up prim type / op selection: Hold Tab\n"
        "* Select prim type / op in selection : LMB\n"
        "* Rotate prim: CTRL+R, move mouse, LMB to confirm\n"
        "* Scale prim: Ctrl+S, move mouse, LMB to confirm\n"
        "* During scaling of prim: \n"
        "   Key 1: scale only X\n"
        "   Key 2: scale only Y\n"
        "   Key 3: scale only Z\n"
        "   Key 4: change op smoothing (no visualization).\n"
        "   Pressing the same key again goes back to scaling all (except Operator Smoothing).\n"
        "\n"
        "Show true primitive preview : Hold Alt + move mouse\n"
        "Add Primitive : Hold Alt + LMB\n"
        "Undo added prim : Ctrl + Z\n"
        "Redo added prim : Ctrl + Y\n"
        "\n"
        "Translate instance : Shift + T, then move mouse or scroll wheel\n"
        "Rotate instance : Shift + R, then move mouse\n"
        "Scale instance : Shift + S, then move mouse\n"
        "\n"
        "Toggle Gridplane : G\n"
        "Toggle Symmetry : H\n"
        "Toggle Primitive Preview : X\n"
        "Toggle Editing on Other Surfaces : C\n"
        "Change Bounding Box Vis : B\n"
        "\n"
        "Rotate grid plane : Hold RMB + move mouse\n"
        "Rotate grid plane around primary axis : Hold CTRL + RMB + move mouse\n"
        "Translate grid plane : Hold Shift + RMB + move mouse\n"
        "\n"
        "Rotate symmetry plane : Hold MMB + move mouse\n"
        "Rotate symmetry plane around primary axis : Hold CTRL + MMB + move mouse\n"
        "Translate symmetry plane : Hold Shift + MMB  + move mouse"
    );

    if (auto group = widget.group("IO", true))
    {
        // TODO: Make the button gray (disabled) instead of removing it.
        if (mCurrentEdit.pSDFGrid->wasInitializedWithPrimitives())
        {
            if (group.button("Save SDF primitives", false))
            {
                std::filesystem::path filePath = "sdfGrid.sdf";
                if (saveFileDialog(kSDFFileExtensionFilters, filePath))
                {
                    mCurrentEdit.pSDFGrid->writePrimitivesToFile(filePath);
                }
            }
        }

        if (group.button("Save SDF grid"))
        {
            std::filesystem::path filePath = "sdfGrid.sdfg";
            if (saveFileDialog(kSDFGridFileExtensionFilters, filePath))
            {
                mCurrentEdit.pSDFGrid->writeValuesFromPrimitivesToFile(filePath, pRenderContext);
            }
        }
    }

    if (auto nodeGroup = widget.group("Grid", true))
    {
        bool gridActive = mUI2D.gridPlane.active > 0 ? true : false;
        widget.checkbox("Show/use grid plane", gridActive);
        mUI2D.gridPlane.active = uint32_t(gridActive);

        if (mUI2D.gridPlane.active)
        {
            // TODO: The limits should ideally be dependent on scene/camera.
            widget.var("Plane center", mUI2D.gridPlane.position, -10.0f, 10.0f, 0.05f);
            widget.direction("Plane normal", mUI2D.gridPlane.normal);
            widget.direction("Plane right vector", mUI2D.gridPlane.rightVector);
            widget.slider("Plane size", mUI2D.gridPlane.planeSize, 0.01f, 2.0f, false, "%2.2f");
            widget.slider("Grid line width", mUI2D.gridPlane.gridLineWidth, 0.01f, 0.1f, false, "%2.2f");
            widget.slider("Grid scale", mUI2D.gridPlane.gridScale, 0.01f, 50.0f, false, "%2.2f");

            widget.rgbaColor("Grid color", mUI2D.gridPlane.color);
        }
    }

    if (auto nodeGroup = widget.group("Bounding Boxes", true))
    {
        static Gui::DropdownList sdfBoundingBoxRenderModes = {
            {uint32_t(SDFBBRenderMode::Disabled), "Disabled"},
            {uint32_t(SDFBBRenderMode::RenderAll), "Render All"},
            {uint32_t(SDFBBRenderMode::RenderSelectedOnly), "Render Selected"},
        };

        widget.dropdown("Render Mode", sdfBoundingBoxRenderModes, mUI2D.bbRenderSettings.renderMode);
        widget.slider("Edge Thickness", mUI2D.bbRenderSettings.edgeThickness, 0.00001f, 0.0005f);
    }

    if (auto group = widget.group("Edit", true))
    {
        if (group.slider<float>("Blobbing", mUI2D.currentBlobbing, kMinShapeBlobbyness, kMaxShapeBlobbyness))
        {
            mCurrentEdit.primitive.shapeBlobbing = mUI2D.currentBlobbing;
        }

        group.var<uint32_t>("Preserved primitives: ", mPreservedHistoryCount, 1);
        group.tooltip("Number of primitives that cannot be baked.");
        group.var<uint32_t>("Batch Size: ", mBakePrimitivesBatchSize, 1);
        group.tooltip("Number of primitives to bake at a time.");
        if (group.button("Bake primitives"))
        {
            mCurrentEdit.pSDFGrid->bakePrimitives(mBakePrimitivesBatchSize);
        }

        group.checkbox("Auto baking", mAutoBakingEnabled);
        group.tooltip("Enable baking at each edit depending on the batch size");

        if (auto innerGroup = group.group("Statistics", true))
        {
            innerGroup.text("#Primitives: " + std::to_string(mCurrentEdit.pSDFGrid->getPrimitiveCount()));
            innerGroup.text("#Baked primitives: " + std::to_string(mCurrentEdit.pSDFGrid->getBakedPrimitiveCount()));
        }
    }
}

bool SDFEditor::onKeyEvent(const KeyboardEvent& keyEvent)
{
    mUI2D.keyboardButtonsPressed.registerCurrentStatesIntoPrevious();

    if (keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        switch (keyEvent.key)
        {
        case Input::Key::LeftControl:
        case Input::Key::RightControl:
            mUI2D.keyboardButtonsPressed.control = true;
            mpScene->setCameraControlsEnabled(false);
            return true;
        case Input::Key::LeftAlt:
        case Input::Key::RightAlt:
            mEditingKeyDown = true;
            mInstanceTransformationEdit.state = TransformationState::None;
            mPrimitiveTransformationEdit.state = TransformationState::None;
            mpScene->setCameraControlsEnabled(false);
            handleToggleEditing();
            return true;
        case Input::Key::LeftShift:
        case Input::Key::RightShift:
            mUI2D.keyboardButtonsPressed.shift = true;
            mPrimitiveTransformationEdit.state = TransformationState::None;
            mpScene->setCameraControlsEnabled(false);
            return true;
        case Input::Key::Tab:
            mGUIKeyDown = true;
            mUI2D.recordStartingMousePos = true;
            mInstanceTransformationEdit.state = TransformationState::None;
            mpScene->setCameraControlsEnabled(false);
            return true;
        case Input::Key::G:
            mUI2D.gridPlane.active = !mUI2D.gridPlane.active;
            return true;
        case Input::Key::H:
            handleToggleSymmetryPlane();
            return true;
        case Input::Key::B:
            mUI2D.bbRenderSettings.renderMode = (mUI2D.bbRenderSettings.renderMode + 1) % uint32_t(SDFBBRenderMode::Count);
            return true;
        case Input::Key::C:
            if (!mEditingKeyDown && !keyEvent.hasModifier(Input::Modifier::Ctrl) && !keyEvent.hasModifier(Input::Modifier::Shift))
            {
                mAllowEditingOnOtherSurfaces = !mAllowEditingOnOtherSurfaces;
                return true;
            }
            break;
        case Input::Key::X:
            if (!mEditingKeyDown && !keyEvent.hasModifier(Input::Modifier::Ctrl) && !keyEvent.hasModifier(Input::Modifier::Shift))
            {
                mPreviewEnabled = !mPreviewEnabled;
                return true;
            }
            break;
        case Input::Key::Z:
            if (keyEvent.hasModifier(Input::Modifier::Ctrl) && !mEditingKeyDown)
            {
                mUndoPressedCount = 0;
                mUI2D.keyboardButtonsPressed.undo = true;
                mUI2D.keyboardButtonsPressed.redo = false;
                handleUndo();
                return true;
            }
            break;
        case Input::Key::Y:
            if (keyEvent.hasModifier(Input::Modifier::Ctrl) && !mEditingKeyDown)
            {
                mRedoPressedCount = 0;
                mUI2D.keyboardButtonsPressed.redo = true;
                mUI2D.keyboardButtonsPressed.undo = false;
                handleRedo();
                return true;
            }
            break;
        case Input::Key::T:
            if (!mEditingKeyDown && keyEvent.hasModifier(Input::Modifier::Shift))
            {
                mInstanceTransformationEdit.state = TransformationState::Translating;
                return true;
            }
            break;
        case Input::Key::R:
            if (!mEditingKeyDown)
            {
                if (keyEvent.hasModifier(Input::Modifier::Shift))
                {
                    mInstanceTransformationEdit.state = TransformationState::Rotating;
                    return true;
                }
                else if (keyEvent.hasModifier(Input::Modifier::Ctrl))
                {
                    mPrimitiveTransformationEdit.state = TransformationState::Rotating;
                    return true;
                }
            }
            break;
        case Input::Key::S:
            if (!mEditingKeyDown)
            {
                if (keyEvent.hasModifier(Input::Modifier::Shift))
                {
                    mInstanceTransformationEdit.state = TransformationState::Scaling;
                    return true;
                }
                else if (keyEvent.hasModifier(Input::Modifier::Ctrl))
                {
                    mPrimitiveTransformationEdit.state = TransformationState::Scaling;
                    return true;
                }
            }
            break;
        case Input::Key::Key1:
            if (mPrimitiveTransformationEdit.state == TransformationState::Scaling)
            {
                mPrimitiveTransformationEdit.axis =
                    mPrimitiveTransformationEdit.axis == SDFEditorAxis::X ? SDFEditorAxis::All : SDFEditorAxis::X;
                return true;
            }
            break;
        case Input::Key::Key2:
            if (mPrimitiveTransformationEdit.state == TransformationState::Scaling)
            {
                mPrimitiveTransformationEdit.axis =
                    mPrimitiveTransformationEdit.axis == SDFEditorAxis::Y ? SDFEditorAxis::All : SDFEditorAxis::Y;
                return true;
            }
            break;
        case Input::Key::Key3:
            if (mPrimitiveTransformationEdit.state == TransformationState::Scaling)
            {
                mPrimitiveTransformationEdit.axis =
                    mPrimitiveTransformationEdit.axis == SDFEditorAxis::Z ? SDFEditorAxis::All : SDFEditorAxis::Z;
                return true;
            }
            break;
        case Input::Key::Key4:
            if (mPrimitiveTransformationEdit.state == TransformationState::Scaling)
            {
                mPrimitiveTransformationEdit.axis =
                    mPrimitiveTransformationEdit.axis == SDFEditorAxis::OpSmoothing ? SDFEditorAxis::All : SDFEditorAxis::OpSmoothing;
                return true;
            }
            break;
        default:
            break;
        }
    }
    else if (keyEvent.type == KeyboardEvent::Type::KeyReleased)
    {
        switch (keyEvent.key)
        {
        case Input::Key::LeftControl:
        case Input::Key::RightControl:
            mUI2D.keyboardButtonsPressed.control = false;
            mUI2D.keyboardButtonsPressed.undo = false;
            mUI2D.keyboardButtonsPressed.redo = false;
            mpScene->setCameraControlsEnabled(true);
            return true;
        case Input::Key::LeftAlt:
        case Input::Key::RightAlt:
            mEditingKeyDown = false;
            mpScene->setCameraControlsEnabled(true);
            handleToggleEditing();
            return true;
        case Input::Key::LeftShift:
        case Input::Key::RightShift:
            mUI2D.keyboardButtonsPressed.shift = false;
            mpScene->setCameraControlsEnabled(true);
            return true;
        case Input::Key::Tab:
            mGUIKeyDown = false;
            mUI2D.recordStartingMousePos = false;
            if (!gridPlaneManipulated() && !symmetryPlaneManipulated() && mPrimitiveTransformationEdit.state == TransformationState::None)
            {
                mUI2D.timeOfReleaseMainGUIKey = mUI2D.timer.getCurrentTimePoint();
                mUI2D.fadeAwayGUI = true;
            }
            mpScene->setCameraControlsEnabled(true);
            return true;
        case Input::Key::Z:
            mUI2D.keyboardButtonsPressed.undo = false;
            return true;
        case Input::Key::Y:
            mUI2D.keyboardButtonsPressed.redo = false;
            return true;
        default:
            break;
        }
    }
    else if (keyEvent.type == KeyboardEvent::Type::KeyRepeated)
    {
        switch (keyEvent.key)
        {
        case Input::Key::Z:
            if (keyEvent.hasModifier(Input::Modifier::Ctrl) && !mEditingKeyDown)
            {
                handleUndo();
                return true;
            }
            return false;
        case Input::Key::Y:
            if (keyEvent.hasModifier(Input::Modifier::Ctrl) && !mEditingKeyDown)
            {
                handleRedo();
                return true;
            }
            return false;
        default:
            break;
        }
    }

    return false;
}

bool SDFEditor::onMouseEvent(const MouseEvent& mouseEvent)
{
    float2 currentMousePos = mouseEvent.pos * float2(mFrameDim);
    mUI2D.currentMousePosition = currentMousePos;

    if (mouseEvent.button == Input::MouseButton::Left)
    {
        mLMBDown = mouseEvent.type == MouseEvent::Type::ButtonDown;
    }

    if (mouseEvent.button == Input::MouseButton::Right)
    {
        mRMBDown = mouseEvent.type == MouseEvent::Type::ButtonDown;
    }

    if (mouseEvent.button == Input::MouseButton::Middle)
    {
        mMMBDown = mouseEvent.type == MouseEvent::Type::ButtonDown;
    }

    bool handled = false;

    if (isMainGUIKeyDown())
    {
        if (mUI2D.recordStartingMousePos)
        {
            mUI2D.recordStartingMousePos = false;
            mUI2D.startMousePosition = currentMousePos;
        }
    }
    else if (mEditingKeyDown && !gridPlaneManipulated())
    {
        if (!mLMBDown)
        {
            if (mouseEvent.type == MouseEvent::Type::Move)
            {
                handleEditMovement();
                handled = true;
            }
        }
        else if ((mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left) || mouseEvent.type == MouseEvent::Type::Move)
        {
            handleAddPrimitive();
            handled = true;
        }
    }
    else if (gridPlaneManipulated())
    {
        if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Right)
        {
            mUI2D.startMousePosition = currentMousePos;
            mUI2D.previousGridPlane = mUI2D.gridPlane;
        }
        else if (mouseEvent.type == MouseEvent::Type::Move)
        {
            // Just pressed shift
            if (mUI2D.keyboardButtonsPressed.shift && !mUI2D.keyboardButtonsPressed.prevShift)
            {
                mUI2D.startMousePosition = mUI2D.currentMousePosition;
                mUI2D.previousGridPlane = mUI2D.gridPlane;
            }
            // Just released shift.
            else if (!mUI2D.keyboardButtonsPressed.shift && mUI2D.keyboardButtonsPressed.prevShift)
            {
                mUI2D.previousGridPlane = mUI2D.gridPlane;
            }

            manipulateGridPlane(
                mUI2D.gridPlane, mUI2D.previousGridPlane, mUI2D.keyboardButtonsPressed.shift, mUI2D.keyboardButtonsPressed.control
            );
        }
    }
    else if (symmetryPlaneManipulated())
    {
        if (mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Middle)
        {
            mUI2D.startMousePosition = currentMousePos;
            mUI2D.previousSymmetryPlane = mUI2D.symmetryPlane;
        }
        else if (mouseEvent.type == MouseEvent::Type::Move)
        {
            // Just pressed shift
            if (mUI2D.keyboardButtonsPressed.shift && !mUI2D.keyboardButtonsPressed.prevShift)
            {
                mUI2D.startMousePosition = mUI2D.currentMousePosition;
                mUI2D.previousSymmetryPlane = mUI2D.symmetryPlane;
            }
            // Just released shift.
            else if (!mUI2D.keyboardButtonsPressed.shift && mUI2D.keyboardButtonsPressed.prevShift)
            {
                mUI2D.previousSymmetryPlane = mUI2D.symmetryPlane;
            }

            manipulateGridPlane(
                mUI2D.symmetryPlane, mUI2D.previousSymmetryPlane, mUI2D.keyboardButtonsPressed.shift, mUI2D.keyboardButtonsPressed.control
            );
        }
    }

    if (mLMBDown)
    {
        mInstanceTransformationEdit.state = TransformationState::None;
        mPrimitiveTransformationEdit.state = TransformationState::None;
    }
    else if (mouseEvent.type == MouseEvent::Type::ButtonUp && mouseEvent.button == Input::MouseButton::Right) // Released right mouse right
                                                                                                              // now.
    {
        mUI2D.previousGridPlane = mUI2D.gridPlane;
    }
    else if (mouseEvent.type == MouseEvent::Type::ButtonUp && mouseEvent.button == Input::MouseButton::Middle) // Released middle mouse
                                                                                                               // right now.
    {
        mUI2D.previousSymmetryPlane = mUI2D.symmetryPlane;
    }
    else if (mouseEvent.type == MouseEvent::Type::Wheel)
    {
        mUI2D.scrollDelta += mouseEvent.wheelDelta.y;

        if (mInstanceTransformationEdit.state != TransformationState::None)
        {
            mInstanceTransformationEdit.scrollTotal += mouseEvent.wheelDelta.y;
        }
    }

    return handled;
}
