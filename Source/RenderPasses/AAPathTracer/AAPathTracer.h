#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"

using namespace Falcor;

class AAPathTracer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(AAPathTracer, "AAPathTracer", "My Path Tracer");

    static ref<AAPathTracer> create(ref<Device> pDevice, const Properties& props) { return make_ref<AAPathTracer>(pDevice, props); }

    AAPathTracer(ref<Device> pDevice, const Properties& props);

    // 获取 RenderPass 属性
    virtual Properties getProperties() const override;
    // 定义 RenderPass 的输入输出通道
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    // 定义 RenderPass 的执行函数
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    // 绘制UI
    virtual void renderUI(Gui::Widgets& widget) override;
    // 设置场景
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    // 鼠标事件
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    // 键盘事件
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    // 解析 RenderPass 属性
    void parseProperties(const Properties& props);
    // 准备着色器变量
    void prepareVars();

    // Internal state
    ref<Scene> mpScene;                     // 当前的场景
    ref<SampleGenerator> mpSampleGenerator; // GPU采样器

    // 渲染参数
    uint mMaxBounces = 3;               // 间接光最大反弹次数
    bool mComputeDirect = true;         // 是否计算直接光照
    bool mUseImportanceSampling = true; // 是否使用重要性采样

    // 运行时数据
    uint mFrameCount = 0;         // 当前帧数
    bool mOptionsChanged = false; // 是否改变了渲染参数

    // Ray tracing 程序
    struct
    {
        ref<RtProgram> pProgram;           // 着色器程序
        ref<RtBindingTable> pBindingTable; // 绑定表
        ref<RtProgramVars> pVars;          // 着色器程序变量
    } mTracer;
};
