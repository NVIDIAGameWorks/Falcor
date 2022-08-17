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
#include "Core/Platform/ProgressBar.h"
#include "Core/Assert.h"

#include <gtk/gtk.h>

#include <chrono>
#include <thread>

namespace Falcor
{
    struct ProgressBarData
    {
        GtkWidget* pWindow = nullptr;
        GtkWidget* pBar = nullptr;
        GtkWidget* pLabel = nullptr;
        // Event IDs/handles
        guint pulseTimerId;
        guint progressUpdateId;

        ProgressBar::MessageList msgList;
        uint32_t msgIndex = 0;

        bool running = true;
        std::thread thread;
    };
    std::unique_ptr<ProgressBarData> ProgressBar::spData;

    ProgressBar::ProgressBar() = default;
    ProgressBar::~ProgressBar()
    {
        close();
    }

    void ProgressBar::close()
    {
        if (spData)
        {
            gtk_window_close(GTK_WINDOW(spData->pWindow));
            spData->running = false;

            g_source_remove(spData->progressUpdateId);
            g_source_remove(spData->pulseTimerId);
            gtk_widget_destroy(spData->pWindow);

            spData->thread.join();
            spData.reset();
        }
    }

    bool ProgressBar::isActive()
    {
        return (bool)spData;
    }

    void progressBarThread(ProgressBarData* pData)
    {
        while(pData->running || gtk_events_pending())
        {
            gtk_main_iteration_do(FALSE);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // At regular intervals, pulse the progress bar
    gboolean progressBarPulseeCB(gpointer pGtkData)
    {
        ProgressBarData* pData = (ProgressBarData*)pGtkData;
        gtk_progress_bar_pulse(GTK_PROGRESS_BAR(pData->pBar));
        return TRUE;
    }

    // At specified intervals, update the window title
    gboolean progressBarUpdateCB(gpointer pGtkData)
    {
        ProgressBarData* pData = (ProgressBarData*)pGtkData;
        if(pData->msgList.size() > 0)
        {
            pData->msgIndex = (pData->msgIndex + 1) % pData->msgList.size();
            gtk_label_set_text(GTK_LABEL(pData->pLabel), pData->msgList[pData->msgIndex].c_str());
        }
        return TRUE;
    }

    void ProgressBar::platformInit(const MessageList& list, uint32_t delayInMs)
    {
        spData.reset(new ProgressBarData);
        spData->msgList = list;

        if (!gtk_init_check(0, nullptr))
        {
            FALCOR_UNREACHABLE();
        }

        // Create window
        spData->pWindow = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_position(GTK_WINDOW(spData->pWindow), GTK_WIN_POS_CENTER_ALWAYS);
        gtk_window_set_decorated(GTK_WINDOW(spData->pWindow), FALSE);

        GtkWidget* pVBox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
        gtk_container_add(GTK_CONTAINER(spData->pWindow), pVBox);

        // Create label for messages
        std::string initialMsg = (list.size() > 0) ? list[0] : "Loading...";
        spData->pLabel = gtk_label_new(initialMsg.c_str());
        gtk_label_set_justify(GTK_LABEL(spData->pLabel), GTK_JUSTIFY_CENTER);
        gtk_label_set_lines(GTK_LABEL(spData->pLabel), 1);
        gtk_box_pack_start(GTK_BOX(pVBox), spData->pLabel, TRUE, FALSE, 0);

        // Create and attach progress bar
        spData->pBar = gtk_progress_bar_new();
        gtk_box_pack_start(GTK_BOX(pVBox), spData->pBar, TRUE, FALSE, 0);

        spData->pulseTimerId = g_timeout_add(100, progressBarPulseeCB, spData.get());
        spData->progressUpdateId = g_timeout_add(delayInMs, progressBarUpdateCB, spData.get());

        gtk_widget_show_all(spData->pWindow);
        spData->thread = std::thread(progressBarThread, spData.get());
    }
}
