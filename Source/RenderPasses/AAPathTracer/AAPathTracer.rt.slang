#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;

// uniform 变量
cbuffer CB{
    uint gFrameCount;       // 当前帧数
    uint gPRNGDimension;    // 第一个可用的 PRNG 的维度
}

// 输入
Texture2D<PackedHitInfo> gVBuffer;  // 编码后的命中的网格体/原始序号和重心坐标
Texture2D<float4> gViewW;           // (可选) 世界空间中的观察方向，在需要景深效果的时候需要

// 输出
RWTexture2D<float4> gOutputColor;   // 输出颜色

// 着色器配置常量, 由cpp程序设置
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

// shadow ray 的 Payload
struct ShadowRayData {
    bool visible;
};

// scatter ray 的 Payload (对齐到72B)
struct ScatterRayData {
    float3 radiance;    // 当前路径的 outgoint radiance
    bool terminated;    // 当前路径终止时为 true
    float3 throughput;  // 当前路径的 throughput, 每个路径顶点都会更新
    uint pathLength;    // 当前路径的段数 (初始为0, 第一次secondary hit时为1, 以此类推, 最大为 2^31)
    float3 origin;      // 下一个路径段的起点
    float3 direction;   // 下一个路径段的方向

    SampleGenerator sg; // 采样器 (对齐到16B)

    // 初始化
    __init(SampleGenerator sg) {
        this.terminated = false;
        this.pathLength = 0;
        this.radiance   = float3(0, 0, 0);
        this.throughput = float3(1, 1, 1);
        this.origin     = float3(0, 0, 0);
        this.direction  = float3(0, 0, 0);
        this.sg         = sg;
    }
};

/** 获取命中点的 ShadingData
    \param[in] hit 命中点信息
    \param[in] rayOrigin 光线起点
    \param[in] rayDir 光线方向
    \param[in] lod 纹理采样器, 计算纹理的分成细节
    \return 命中点的 ShadingData
*/
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const ITextureSampler lod) {
    VertexData v = {};
    uint materialID = {};

    // 场景中含有 triangle mesh 类型的几何体
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle) {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif

    // 场景中含有 displaced triangle mesh 类型的几何体
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle) {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif

    // 场景中含有 curve 类型的几何体
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve) {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif

    // 场景中含有 SDF grid 类型的几何体
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid) {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir, lod);
    return sd;
}

/** 获取 primary ray 的方向
    \param[in] launchIndex 当前像素的位置
    \param[in] launchDim 当前帧的维度
    \param[in] camera 相机
    \return primary ray 的方向
*/
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera) {
    // 如果有观察方向, 则返回观察方向
    if (is_valid(gViewW)) {
        return -gViewW[launchIndex].xyz;
    }
    // 否则计算观察方向
    else {
        return camera.computeRayPinhole(launchIndex, launchDim).dir;
    }
}

/** 追踪 shadow ray: 当前点与光源的连线
    \param[in] origin 光线起点
    \param[in] dir 光线方向
    \param[in] distance 当前点与光源的距离
    \return True: 该光源可见
*/
bool traceShadowRay(float3 origin, float3 dir, float distance) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false;
    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xff, 1, rayTypeCount, 1, ray, rayData);

    return rayData.visible;
}

/** 追踪 scatter ray: 当前点散射出去的光线
    \param[inout] rayData scatter ray的数据, 会根据计算结果更新
*/
void traceScatterRay(inout ScatterRayData rayData) {
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0;
    TraceRay(gScene.rtAccel, rayFlags, 0xff, 0, rayTypeCount, 0, ray, rayData);
}

/** 计算直接光照
    \param[in] shadingData 命中点的 ShadingData
    \param[in] materialInstance 命中点的 MaterialInstance
    \param[inout] sampleGenerator 采样器
    \return 观察方向的 outgoing radiance
*/
float3 evalDirectAnalytic(const ShadingData shadingData, const IMaterialInstance materialInstance, inout SampleGenerator sampleGenerator) {
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0) return float3(0.f);

    // 随机选择一个光源
    const uint lightIndex = min(uint(sampleNext1D(sampleGenerator) * lightCount), lightCount - 1);
    // 选中的概率为 PDF = 1 / lightCount
    float invPDF = lightCount;

    // 采样 local light source
    AnalyticLightSample lightSample;
    if (!sampleLight(shadingData.posW, gScene.getLight(lightIndex), sampleGenerator, lightSample)) return float3(0.f);

    // 如果采样点不在BSDF的半球内, 则忽略该采样点
    const uint lobeTypes = materialInstance.getLobeTypes(shadingData);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTrasmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(shadingData.getOrientedFaceNormal(), lightSample.dir);
    if((NdotL <= kMinCosTheta && !hasTrasmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // 获取在几何体法线方向应用偏移的起点，以避免自相交
    const float3 origin = computeRayOrigin(shadingData.posW, dot(shadingData.faceN, lightSample.dir) >= 0.f ? shadingData.faceN : -shadingData.faceN);

    // 通过追踪 shadow ray 来判断该光源是否可见
    bool V = traceShadowRay(origin, lightSample.dir, lightSample.distance);
    if(!V) return float3(0.f);

    // 计算贡献
    return materialInstance.eval(shadingData, lightSample.dir, sampleGenerator) * lightSample.Li * invPDF;
}

/** 生成一个新的 scatter ray, 或者终止当前路径
    \param[in] shadingData 命中点的 ShadingData
    \param[in] materialInstance 命中点的 MaterialInstance
    \param[in] isCurveHit 是否是 curve hit
    \param[in] rayOrigin 新的 ray 的起点
    \param[inout] rayData scatter ray的数据, 会根据计算结果更新
    \return True: 当前路径继续
*/
bool generateScatterRay(const ShadingData shadingData, const IMaterialInstance materialInstance, bool isCurveHit, float3 rayOrigin, inout ScatterRayData rayData){
    // 依据BSDF进行采样
    BSDFSample bsdfSample;
    if (materialInstance.sample(shadingData, rayData.sg, bsdfSample, kUseImportanceSampling)) {
        rayData.origin = rayOrigin;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission)) {
            rayData.origin = shadingData.computeNewRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;
        rayData.throughput *= bsdfSample.weight;
        return any(rayData.throughput > 0.f);
    }

    return false;
}

/** 处理一次hit: 加载shading data, 采样 analytic lights, 采样新的scatter ray, 如果达到最大反射次数则终止路径
    \param[in] hit 命中点信息
    \param[inout] rayData scatter ray的数据, 会根据计算结果更新
*/
void handleHit(const HitInfo hit, inout ScatterRayData rayData) {
    // 判断是否是 curve hit
    const bool isCurveHit = hit.getType() == HitType::Curve;

    // 加载 lod 纹理
    let lod = ExplicitLodTextureSampler(0.f);

    // 加载 shading data
    ShadingData shadingData = loadShadingData(hit, rayData.origin, rayData.direction, lod);

    // 创造 material instance
    let materialInstance = gScene.materials.getMaterialInstance(shadingData, lod);

    // 添加自发光
    if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0)) {
        rayData.radiance += rayData.throughput * materialInstance.getProperties(shadingData).emission;
    }

    // 基于最大深度, 检查是否终止路径
    if (rayData.pathLength >= kMaxBounces) {
        rayData.terminated = true;
        return;
    }

    // 计算从当前点生成的新的 ray 的起点
    float3 rayOrigin;
    if (isCurveHit) {
        // 对于 curves, 新的起点为球心
        rayOrigin = shadingData.posW - shadingData.curveRadius * shadingData.frame.N;
    } else {
        rayOrigin = shadingData.computeNewRayOrigin();
    }

    // 添加来自 analytic lights 的直接光照
    if (kUseAnalyticLights) {
        float3 Lr = evalDirectAnalytic(shadingData, materialInstance, rayData.sg);
        rayData.radiance += rayData.throughput * Lr;
    }

    // 生成新的 scatter ray
    if (!generateScatterRay(shadingData, materialInstance, isCurveHit, rayOrigin, rayData)) {
        rayData.terminated = true;
        return;
    }

    rayData.pathLength++;
}

/** 进行路径追踪
    \param[in] pixel 当前像素的位置
    \param[in] frameDim 当前帧的维度
    \return 路径追踪得到的颜色
*/
float3 tracePath(const uint2 pixel, const uint2 frameDim) {
    float3 outColor = float3(0.f);

    // 生成 primary ray
    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);

    // 碰撞点信息
    const HitInfo hit = HitInfo(gVBuffer[pixel]);

    // 碰撞有效, 进行路径追踪
    if (hit.isValid()) {
        // 判断是否是 curve hit
        const bool isCurveHit = hit.getType() == HitType::Curve;

        // 获取纹理的 lod
        let lod = ExplicitLodTextureSampler(0.f);

        // 获取碰撞点的 shading data
        ShadingData shadingData = loadShadingData(hit, primaryRayOrigin, primaryRayDir, lod);
        // 获取碰撞点的 material instance
        let materialInstance = gScene.materials.getMaterialInstance(shadingData, lod);

        // 创建 sample generator
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        for (uint i = 0; i < gPRNGDimension; i++) sampleNext1D(sg);

        // 计算光线的起点
        float3 rayOrigin;
        if (isCurveHit) {
            // 对于 curves, 新的起点为球心
            rayOrigin = shadingData.posW - shadingData.curveRadius * shadingData.frame.N;
        } else {
            rayOrigin = shadingData.computeNewRayOrigin();
        }

        // 计算直接光照
        if (kComputeDirect) {
            // 计算当前点的自发光
            outColor += materialInstance.getProperties(shadingData).emission;

            // 添加光源的直接光照
            outColor += kUseAnalyticLights ? evalDirectAnalytic(shadingData, materialInstance, sg) : float3(0.f);
        }

        // 准备 ray payload
        ScatterRayData rayData = ScatterRayData(sg);

        // 生成 scatter ray
        if (!generateScatterRay(shadingData, materialInstance, isCurveHit, rayOrigin, rayData))
            rayData.terminated = true;

        // 追踪 scatter ray, 最多 kMaxBounces 次
        for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
            traceScatterRay(rayData);

        outColor += rayData.radiance;
    }
    // 碰撞无效, 使用天空盒的颜色
    else {
        outColor = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
    }

    return outColor;
}


/* 未击中的 shader 入口  */
[shader("miss")]
void scatterMiss(inout ScatterRayData rayData) {
    rayData.terminated = true;

    // 计算天空盒的颜色
    if (kUseEnvLight && (kComputeDirect || rayData.pathLength > 0)) {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += rayData.throughput * Le;
    }
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData) {
    rayData.visible = true;
}


/* 击中 TriangleMesh 的 shader 入口 */
[shader("anyhit")]
void scatterTriangleMeshAnyHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs) {
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    // 对于不透明物体, 进行alpha检测
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

[shader("closesthit")]
void scatterTriangleMeshClosestHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs) {
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(triangleHit), rayData);
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs) {
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    // 进行 alpha test
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

/* 击中 DisplacedTriangleMesh 物体的 shader 入口 */
[shader("intersection")]
void displacedTriangleMeshIntersection() {
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
        ReportHit(t, 0, attribs);
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout ScatterRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs) {
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    handleHit(HitInfo(displacedTriangleHit), rayData);
}

/* 击中 Curve 物体的 shader 入口 */
[shader("intersection")]
void curveIntersection() {
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
        ReportHit(t, 0, attribs);
}

[shader("closesthit")]
void scatterCurveClosestHit(inout ScatterRayData rayData, CurveIntersector::Attribs attribs) {
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(curveHit), rayData);
}

/* 击中 SDFGrid 物体的 shader 入口 */
[shader("intersection")]
void sdfGridIntersection() {
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
        ReportHit(t, 0, sdfGridHitData);
}

[shader("closesthit")]
void scatterSDFGridClosestHit(inout ScatterRayData rayData, SDFGridHitData sdfGridHitData) {
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    handleHit(HitInfo(sdfGridHit), rayData);
}

/* 光线生成的 shader 入口*/
[shader("raygeneration")]
void rayGen() {
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    float3 color = tracePath(pixel, frameDim);

    gOutputColor[pixel] = float4(color, 1.f);
}
