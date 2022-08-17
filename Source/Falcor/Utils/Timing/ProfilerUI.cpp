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
#include "ProfilerUI.h"
#include "Core/Platform/OS.h"

#include <imgui.h>

namespace Falcor
{
    namespace
    {
        const char* kGraphModes[(size_t)ProfilerUI::GraphMode::Count] = { "Off", "CPU Time", "GPU Time" };

        const float kIndentWidth = 16.f;
        const float kPadding = 16.f;
        const float kBarWidth = 50.f;
        const float kBarHeight = 8.f;
        const uint32_t kBarColor = 0xffffffff;
        const uint32_t kBarMutedColor = 0xff404040;
        const float kGraphBarWidth = 2.f;

        static constexpr size_t kColumnCount = 5;
        const char* kColumnTitle[kColumnCount] = { "Event", "CPU Time", "CPU %%", "GPU Time", "GPU %%" };
        const float kHeaderSpacing = 5.f;

        const size_t kHistoryCapacity = 256;

        // Colorblind friendly palette.
        const std::vector<uint32_t> kColorPalette = {
            IM_COL32(0x00, 0x49, 0x49, 0xff),
            IM_COL32(0x00, 0x92, 0x92, 0xff),
            IM_COL32(0xff, 0x6d, 0xb6, 0xff),
            IM_COL32(0xff, 0xb6, 0xdb, 0xff),
            IM_COL32(0x49, 0x00, 0x92, 0xff),
            IM_COL32(0x00, 0x6d, 0xdb, 0xff),
            IM_COL32(0xb6, 0x6d, 0xff, 0xff),
            IM_COL32(0x6d, 0xb6, 0xff, 0xff),
            IM_COL32(0xb6, 0xdb, 0xff, 0xff),
            IM_COL32(0x92, 0x00, 0x00, 0xff),
            // Yellow-ish colors don't work well with the highlight color.
            // IM_COL32(0x92, 0x49, 0x00, 0xff),
            // IM_COL32(0xdb, 0x6d, 0x00, 0xff),
            IM_COL32(0x24, 0xff, 0x24, 0xff),
            IM_COL32(0xff, 0xff, 0x6d, 0xff)
        };

        const uint32_t kHighlightColor = IM_COL32(0xff, 0x7f, 0x00, 0xcf);

        void drawRectFilled(const ImVec2& pos, const ImVec2& size, uint32_t color = 0xffffffff)
        {
            auto cursorPos = ImGui::GetCursorScreenPos();
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(cursorPos.x + pos.x, cursorPos.y + pos.y), ImVec2(cursorPos.x + pos.x + size.x, cursorPos.y + pos.y + size.y), color);
        };

        void drawBar(float fraction, const ImVec2& size, ImU32 color = 0xffffffff, ImU32 background = 0x00000000, bool highlight = false)
        {
            auto cursorPos = ImGui::GetCursorScreenPos();
            auto height = ImGui::GetTextLineHeightWithSpacing();
            cursorPos.y += 0.5f * (height - size.y);
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(cursorPos.x, cursorPos.y), ImVec2(cursorPos.x + size.x, cursorPos.y + size.y), background);
            ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(cursorPos.x, cursorPos.y), ImVec2(cursorPos.x + fraction * size.x, cursorPos.y + size.y), color);
            if (highlight) ImGui::GetWindowDrawList()->AddRect(ImVec2(cursorPos.x, cursorPos.y), ImVec2(cursorPos.x + size.x, cursorPos.y + size.y), kHighlightColor);
        }
    }

    ProfilerUI::UniquePtr ProfilerUI::create(const Profiler::SharedPtr& pProfiler)
    {
        return UniquePtr(new ProfilerUI(pProfiler));
    }

    void ProfilerUI::render()
    {
        updateEventData();
        updateGraphData();

        renderOptions();

        // Compute column widths.
        float columnWidth[kColumnCount];
        for (size_t i = 0; i < kColumnCount; ++i) columnWidth[i] = ImGui::CalcTextSize(kColumnTitle[i]).x + kPadding;
        for (const auto& eventData : mEventData)
        {
            columnWidth[0] = std::max(columnWidth[0], ImGui::CalcTextSize(eventData.name.c_str()).x + eventData.level * kIndentWidth + kPadding);
        }

        float startY;
        float endY;
        size_t newHighlightIndex = -1;

        // Draw table (last column is used for graph).
        ImGui::Columns(kColumnCount + 1, "#events", false);
        for (size_t col = 0; col < kColumnCount; ++col)
        {
            ImGui::SetColumnWidth((int)col, columnWidth[col]);
            ImGui::TextUnformatted(kColumnTitle[col]);
            ImGui::Dummy(ImVec2(0.f, kHeaderSpacing));

            if (col == 0) startY = ImGui::GetCursorPosY();

            for (size_t i = 0; i < mEventData.size(); ++i)
            {
                const auto& eventData = mEventData[i];
                auto pEvent = eventData.pEvent;

                if (col == 0) // Event name
                {
                    float indent = eventData.level * kIndentWidth;
                    if (indent > 0.f) ImGui::Indent(indent);
                    auto color = (i == mHighlightIndex) ? ImColor(kHighlightColor) : ImColor(ImGui::GetStyleColorVec4(ImGuiCol_Text));
                    ImGui::TextColored(color, "%s", eventData.name.c_str());
                    if (indent > 0.f) ImGui::Unindent(indent);
                }
                else if (col == 1) // CPU time
                {
                    ImGui::Text("%.2f ms", eventData.cpuTime);
                    if (ImGui::IsItemHovered())
                    {
                        auto stats = eventData.pEvent->computeCpuTimeStats();
                        ImGui::BeginTooltip();
                        ImGui::Text("%s\nMin: %.2f\nMax: %.2f\nMean: %.2f\nStdDev: %.2f", eventData.name.c_str(), stats.min, stats.max, stats.mean, stats.stdDev);
                        ImGui::EndTooltip();
                    }
                }
                else if (col == 2) // CPU %
                {
                    ImGui::PushID((int)reinterpret_cast<intptr_t>(&eventData));
                    float fraction = mTotalCpuTime > 0.f ? eventData.cpuTime / mTotalCpuTime : 0.f;
                    bool isGraphShown = mGraphMode == GraphMode::CpuTime;
                    bool isHighlighted = isGraphShown && i == mHighlightIndex;
                    drawBar(fraction, ImVec2(kBarWidth, kBarHeight), isGraphShown ? eventData.color : kBarColor, isGraphShown ? eventData.mutedColor : kBarMutedColor, isHighlighted);
                    ImGui::Dummy(ImVec2(kBarWidth, ImGui::GetTextLineHeight()));
                    if (ImGui::IsItemHovered())
                    {
                        if (isGraphShown) newHighlightIndex = i;
                        ImGui::BeginTooltip();
                        ImGui::Text("%s\n%.1f%%", eventData.name.c_str(), fraction * 100.f);
                        ImGui::EndTooltip();
                    }
                    ImGui::PopID();
                }
                else if (col == 3) // GPU time
                {
                    ImGui::Text("%.2f ms", eventData.gpuTime);
                    if (ImGui::IsItemHovered())
                    {
                        auto stats = eventData.pEvent->computeGpuTimeStats();
                        ImGui::BeginTooltip();
                        ImGui::Text("%s\nMin: %.2f\nMax: %.2f\nMean: %.2f\nStdDev: %.2f", eventData.name.c_str(), stats.min, stats.max, stats.mean, stats.stdDev);
                        ImGui::EndTooltip();
                    }
                }
                else if (col == 4) // GPU %
                {
                    ImGui::PushID((int)reinterpret_cast<intptr_t>(&eventData));
                    float fraction = mTotalGpuTime > 0.f ? eventData.gpuTime / mTotalGpuTime : 0.f;
                    bool isGraphShown = mGraphMode == GraphMode::GpuTime;
                    bool isHighlighted = isGraphShown && i == mHighlightIndex;
                    drawBar(fraction, ImVec2(kBarWidth, kBarHeight), isGraphShown ? eventData.color : kBarColor, isGraphShown ? eventData.mutedColor : kBarMutedColor, isHighlighted);
                    ImGui::Dummy(ImVec2(kBarWidth, ImGui::GetTextLineHeight()));
                    if (ImGui::IsItemHovered())
                    {
                        if (isGraphShown) newHighlightIndex = i;
                        ImGui::BeginTooltip();
                        ImGui::Text("%s\n%.1f%%", eventData.name.c_str(), fraction * 100.f);
                        ImGui::EndTooltip();
                    }
                    ImGui::PopID();
                }
            }

            if (col == 0) endY = ImGui::GetCursorPosY();

            ImGui::NextColumn();
        }

        // Set new highlight index if mouse is over one of the bars.
        if (newHighlightIndex != -1) mHighlightIndex = newHighlightIndex;

        // Draw the graph.
        if (mGraphMode != GraphMode::Off)
        {
            ImGui::Text("Graph");
            ImGui::Dummy(ImVec2(0.f, kHeaderSpacing));
            ImVec2 graphSize(ImGui::GetWindowSize().x - ImGui::GetCursorPosX(), endY - startY);
            renderGraph(graphSize, mHighlightIndex, newHighlightIndex);
            mHighlightIndex = newHighlightIndex;
        }
    }

    void ProfilerUI::renderOptions()
    {
        bool paused = mpProfiler->isPaused();
        if (ImGui::Checkbox("Pause", &paused)) mpProfiler->setPaused(paused);

        ImGui::SameLine();
        ImGui::Checkbox("Average", &mEnableAverage);

        ImGui::SameLine();
        ImGui::SetNextItemWidth(100.f);
        if (ImGui::Combo("Graph", reinterpret_cast<int*>(&mGraphMode), kGraphModes, (int)GraphMode::Count)) clearGraphData();

        if (mpProfiler->isCapturing())
        {
            ImGui::SameLine();
            if (ImGui::Button("End Capture"))
            {
                auto pCapture = mpProfiler->endCapture();
                FALCOR_ASSERT(pCapture);
                FileDialogFilterVec filters {{ "json", "JSON" }};
                std::filesystem::path path;
                if (saveFileDialog(filters, path))
                {
                    pCapture->writeToFile(path);
                }
            }
        }
        else
        {
            ImGui::SameLine();
            if (ImGui::Button("Start Capture")) mpProfiler->startCapture();
        }

        ImGui::Separator();
    }

    void ProfilerUI::renderGraph(const ImVec2& size, size_t highlightIndex, size_t& newHighlightIndex)
    {
        ImVec2 mousePos = ImGui::GetMousePos();
        ImVec2 screenPos = ImGui::GetCursorScreenPos();
        mousePos.x -= screenPos.x;
        mousePos.y -= screenPos.y;

        float totalMaxGraphValue = 0.f;
        for (const auto& eventData : mEventData) totalMaxGraphValue += eventData.level == 0 ? eventData.maxGraphValue : 0.f;

        const float scaleY = size.y / totalMaxGraphValue;

        float x = 0.f;
        float levelY[128];
        std::optional<float> highlightValue;

        for (size_t k = 0; k < mHistoryLength; ++k)
        {
            size_t historyIndex = (mHistoryWrite + kHistoryCapacity - k - 1) % kHistoryCapacity;

            float totalValue = 0.f;
            for (const auto& eventData : mEventData)
            {
                totalValue += eventData.level == 0 ? eventData.graphHistory[historyIndex] : 0.f;
            }

            levelY[0] = size.y - totalValue * scaleY;
            float highlightY = 0.f;
            float highlightHeight = 0.f;

            for (size_t i = 0; i < mEventData.size(); ++i)
            {
                const auto& eventData = mEventData[i];
                float value = eventData.graphHistory[historyIndex];
                uint32_t level = eventData.level;

                float y = levelY[level];
                float height = value * scaleY;
                drawRectFilled(ImVec2(x, y), ImVec2(kGraphBarWidth, height), eventData.color);

                // Check if mouse is over this bar for tooltip value and highlighting in next frame.
                if (mousePos.x >= x && mousePos.x < x + kGraphBarWidth && mousePos.y >= y && mousePos.y < y + height)
                {
                    newHighlightIndex = i;
                    highlightValue = value / totalValue;
                }

                if (highlightIndex == i)
                {
                    highlightY = y;
                    highlightHeight = height;
                }

                levelY[level + 1] = levelY[level];
                levelY[level] += height;
            }

            if (highlightHeight > 0.f) drawRectFilled(ImVec2(x, highlightY), ImVec2(kGraphBarWidth, highlightHeight), kHighlightColor);

            x += kGraphBarWidth;
            if (x > size.x) break;
        }

        ImGui::Dummy(ImVec2(size));
        if (ImGui::IsItemHovered() && highlightValue)
        {
            FALCOR_ASSERT(newHighlightIndex >= 0 && newHighlightIndex < mEventData.size());
            ImGui::BeginTooltip();
            ImGui::Text("%s\n%.2f%%", mEventData[newHighlightIndex].name.c_str(), *highlightValue * 100.f);
            ImGui::EndTooltip();
        }
    }

    void ProfilerUI::updateEventData()
    {
        const auto& events = mpProfiler->getEvents();

        mEventData.resize(events.size());
        mTotalCpuTime = 0.f;
        mTotalGpuTime = 0.f;

        for (size_t i = 0; i < mEventData.size(); ++i)
        {
            auto& event = mEventData[i];
            auto pEvent = events[i];

            event.pEvent = pEvent;

            // Update name and level.
            std::string name = pEvent->getName();
            uint32_t level = std::max((uint32_t)std::count(name.begin(), name.end(), '/'), 1u) - 1;
            name = name.substr(name.find_last_of("/") + 1);
            event.name = name;
            event.level = level;

            // Use colors from color palette.
            event.color = kColorPalette[i % kColorPalette.size()];
            event.mutedColor = (event.color & 0xffffff) | 0x1f000000;

            // Get event times.
            event.cpuTime = mEnableAverage ? pEvent->getCpuTimeAverage() : pEvent->getCpuTime();
            event.gpuTime = mEnableAverage ? pEvent->getGpuTimeAverage() : pEvent->getGpuTime();

            // Sum up times.
            if (level == 0)
            {
                mTotalCpuTime += event.cpuTime;
                mTotalGpuTime += event.gpuTime;
            }
        }
    }

    void ProfilerUI::updateGraphData()
    {
        if (mGraphMode == GraphMode::Off) return;

        mTotalGraphValue = 0.f;

        for (auto& event : mEventData)
        {
            switch (mGraphMode)
            {
            case GraphMode::Off:
                continue;
            case GraphMode::CpuTime:
                event.graphValue = event.cpuTime;
                break;
            case GraphMode::GpuTime:
                event.graphValue = event.gpuTime;
                break;
            case GraphMode::Count:
                break;
            }

            if (event.level == 0) mTotalGraphValue += event.graphValue;

            event.graphHistory.resize(kHistoryCapacity);
            event.graphHistory[mHistoryWrite] = event.graphValue;

            float maxGraphValue = 0.f;
            for (size_t j = 0; j < mHistoryLength; ++j)
            {
                maxGraphValue = std::max(maxGraphValue, event.graphHistory[j]);
            }
            event.maxGraphValue = maxGraphValue;
        }

        if (!mpProfiler->isPaused())
        {
            mHistoryWrite = (mHistoryWrite + 1) % kHistoryCapacity;
            mHistoryLength = std::min(mHistoryLength + 1, kHistoryCapacity);
        }
    }

    void ProfilerUI::clearGraphData()
    {
        mHistoryLength = 0;
        mHistoryWrite = 0;

        for (auto& event : mEventData)
        {
            event.graphValue = 0.f;
            event.maxGraphValue = 0.f;
        }
    }
}
