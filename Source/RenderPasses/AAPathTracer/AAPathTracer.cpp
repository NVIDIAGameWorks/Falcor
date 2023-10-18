#include "AAPathTracer.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry) {
    registry.registerClass<RenderPass, AAPathTracer>();
}

namespace {
    // 着色器程序文件路径
    const char kShaderFile[] = "RenderPasses/AAPathTracer/AAPathTracer.rt.slang";

    // Ray tracing 设置, 影响追踪的栈的大小
    const uint32_t kMaxPayloadSizeBytes = 72u;  // 最大 payload 大小
    const uint32_t kMaxRecursionDepth = 2u;     // 最大递归深度

    // View Dir
    const char kInputViewDir[] = "viewW";

    // 输入通道: vbuffer, viewW
    const ChannelList kInputChannels = {
        // 编码后的命中的网格体/原始序号和重心坐标
        {"vbuffer", "gVBuffer", "Visibility buffer in packed format"},

        // (可选) 世界空间中的观察方向，在需要景深效果的时候需要
        {kInputViewDir, "gViewW", "World-space view direction (xyz float format)", true},
    };

    // 输出通道: color
    const ChannelList kOutputChannels = {
        // 输出颜色, 为RGBA32Float格式
        {"color", "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float},
    };

    // 最大反弹次数
    const char kMaxBounces[] = "maxBounces";
    // 是否计算直接光照
    const char kComputeDirect[] = "computeDirect";
    // 是否使用重要性采样
    const char kUseImportanceSampling[] = "useImportanceSampling";

} // namespace

// 绘制UI
void AAPathTracer::renderUI(Gui::Widgets& widget) {
    bool dirty = false;

    // 最大反弹次数
    dirty |= widget.var("Max Bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    // 是否计算直接光照
    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    // 是否使用重要性采样
    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    // 如果 options 更改了, 那么要设置对应的 flag, 在 execute() 中会检测该 flag
    if (dirty) mOptionsChanged = true;
}

AAPathTracer::AAPathTracer(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    // 解析属性
    parseProperties(props);

    // 创建sample generator
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

// 解析 RenderPass 属性
void AAPathTracer::parseProperties(const Properties& props) {
    for (const auto& [key, value] : props) {
        if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else
            logWarning("Unknown property '{}' in MinimalPathTracer properties.", key);
    }
}

// 获取 RenderPass 属性
Properties AAPathTracer::getProperties() const {
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

// 定义 RenderPass 的输入输出通道
RenderPassReflection AAPathTracer::reflect(const CompileData& compileData) {
    RenderPassReflection reflector;

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

// 定义 RenderPass 的执行函数
void AAPathTracer::execute(RenderContext* pRenderContext, const RenderData& renderData) {
    // 获取所有渲染层的共享资源
    auto& dict = renderData.getDictionary();

    // 如果 UI 上的 options 发生了改变, 则需要将 RenderPassRefreshFlags 添加上 RenderOptionsChanged
    if (mOptionsChanged) {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // 如果没有场景, 则清空输出并返回
    if (!mpScene) {
        for (auto it : kOutputChannels) {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst) pRenderContext->clearTexture(pDst);
        }
        return;
    }

    // 不支持场景几何信息发生改变
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
        throw RuntimeError("AAPathTracer: This render pass does not support scene geometry changes.");

    // 如果场景中有自发光, 则获取 light collection
    if (mpScene->getRenderSettings().useEmissiveLights)
        mpScene->getLightCollection(pRenderContext);

    // 相机光圈半径大于0, 则开启景深(Depth-of-field)
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);

    // 配置着色器程序常量
    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");

    // 对于可选的 I/O 资源, 设置 'is_valid_<name>' 定义, 告知着色器程序哪些资源是有效的
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // 准备着色器变量. 在这之前, 着色器程序必须拥有所有必要的定义
    if (!mTracer.pVars) prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // 设置着色器 uniform 变量
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;

    // 绑定 I/O buffers. 这需要在每帧都进行一次, 因为 buffers 随时都可能改变
    for (auto channel : kInputChannels)
        if(!channel.texname.empty())
            var[channel.texname] = renderData.getTexture(channel.name);
    for (auto channel : kOutputChannels)
        if(!channel.texname.empty())
            var[channel.texname] = renderData.getTexture(channel.name);

    // 获取 ray dispatch 的维度
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // 生成 ray
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));

    // 帧数++
    mFrameCount++;
}

// 设置场景
void AAPathTracer::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) {
    // 清除原有场景的数据
    // 更换场景后, ray tracing 程序需要重新创建
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // 设置新的场景
    mpScene = pScene;

    if (mpScene) {
        // 不支持自定义类型的几何体
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
            logWarning("AAPathTracer: This render pass does not support custom primitives.");


        // 创建 ray tracing 程序
        RtProgram::Desc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        // 创建 binding table
        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto shaderBindingTable = mTracer.pBindingTable;
        shaderBindingTable->setRayGen(desc.addRayGen("rayGen"));
        shaderBindingTable->setMiss(0, desc.addMiss("scatterMiss"));
        shaderBindingTable->setMiss(1, desc.addMiss("shadowMiss"));

        // 场景中有 TriangleMesh 类型的几何体
        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh)) {
            shaderBindingTable->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit"));
            shaderBindingTable->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit"));
        }

        // 场景中有 DisplacedTriangleMesh 类型的几何体
        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh)) {
            shaderBindingTable->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh), desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection"));
            shaderBindingTable->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh), desc.addHitGroup("", "", "displacedTriangleMeshIntersection"));
        }

        // 场景中有 Curve 类型的几何体
        if (mpScene->hasGeometryType(Scene::GeometryType::Curve)) {
            shaderBindingTable->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection"));
            shaderBindingTable->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("", "", "curveIntersection"));
        }

        // 场景中有 SDFGrid 类型的几何体
        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid)) {
            shaderBindingTable->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection"));
            shaderBindingTable->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("", "", "sdfGridIntersection"));
        }

        // 创建着色器程序
        mTracer.pProgram = RtProgram::create(mpDevice, desc, mpScene->getSceneDefines());
    }
}

// 准备着色器变量
void AAPathTracer::prepareVars() {
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // 配置着色器程序
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // 创建着色器变量
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // 绑定 SampleGenerator 到 共享数据
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->setShaderData(var);
}
