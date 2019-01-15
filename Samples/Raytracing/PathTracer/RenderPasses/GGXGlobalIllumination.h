#pragma once
#include "Falcor.h"
#include "FalcorExperimental.h"

using namespace Falcor;

/** This pass defines the multi-bounce global illumination pass in Chris' "Introduction to Direct Raytracing"
    tutorials
*/
class GGXGlobalIllumination : public RenderPass, inherit_shared_from_this<RenderPass, GGXGlobalIllumination>
{
public:
    using SharedPtr = std::shared_ptr<GGXGlobalIllumination>;

    /** Instantiate our pass.  The input Python dictionary is where you can extract pass parameters
    */
    static SharedPtr create(const Dictionary& params = {});
    
    /** Get a string describing what the pass is doing
    */
    virtual std::string getDesc() override { return "Path traces from a G-buffer to accumulate indirect light using a GGX lighting model."; }

    /** Defines the inputs/outputs required for this render pass
    */
    virtual RenderPassReflection reflect(void) const override;

    /** Run our multibounce GI pass
    */
    virtual void execute(RenderContext* pContext, const RenderData* pRenderData) override;

    /** Display a GUI exposing rendering parameters
    */
    virtual void renderUI(Gui* pGui, const char* uiGroup) override;

    /** Grab the current scene so we can render it!
    */
    virtual void setScene(const std::shared_ptr<Scene>& pScene) override;

    /** Do any updates needed when we resize our window
    */
    virtual void onResize(uint32_t width, uint32_t height) override;

    /** Serialize the render pass parameters out to a python dictionary
    */
    virtual Dictionary getScriptingDictionary() const override;

private:
    GGXGlobalIllumination() : RenderPass("GGXGlobalIllumination") {}

    /** Runs on first execute() to initialize rendering resources
    */
    void initialize(RenderContext* pContext, const RenderData* pRenderData);

    // Internal pass state
    RtProgram::SharedPtr        mpProgram;
    RtState::SharedPtr          mpState;
    RtProgramVars::SharedPtr    mpVars;

    RtScene::SharedPtr          mpScene;
    RtSceneRenderer::SharedPtr  mpSceneRenderer;

    // Recursive ray tracing can be slow. Add a toggle to disable, to allow you to manipulate the scene
    bool                mDoIndirectGI = true;
    bool                mDoDirectGI = true;
    bool                mUseEmissiveGeom = true;
    float               mEmissiveGeomMult = 1.0f;
    Texture::SharedPtr  mpBlackHDR = nullptr;

    enum class EnvMapMode : uint32_t
    {
        Scene,
        Black
    };

    EnvMapMode mEnvMapMode = EnvMapMode::Scene;

    int32_t         mUserSpecifiedRayDepth = 2; ///<  What is the current maximum ray depth
    const int32_t   mMaxPossibleRayDepth = 4;   ///<  The largest ray depth we support (without recompile)

    // A frame counter; needs to start at a different value than passes, since it uses to seed the RNG
    uint32_t mFrameCount = 0x1234u; ///< A frame counter to vary random numbers over time

    // Some common pass bookkeeping
    uvec3   mRayLaunchDims = uvec3(0, 0, 0);
    bool    mIsInitialized = false;
};
