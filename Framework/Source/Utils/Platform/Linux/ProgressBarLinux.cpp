/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/

#include "Framework.h"
#include "Utils/Platform/ProgressBar.h"
#include <chrono>
#include <gtk/gtk.h>

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

    ProgressBar::~ProgressBar()
    {
        gtk_window_close(GTK_WINDOW(mpData->pWindow));
        mpData->running = false;

        g_source_remove(mpData->progressUpdateId);
        g_source_remove(mpData->pulseTimerId);
        gtk_widget_destroy(mpData->pWindow);

        mpData->thread.join();
        safe_delete(mpData);
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
        mpData = new ProgressBarData;
        mpData->msgList = list;

        if (!gtk_init_check(0, nullptr))
        {
            should_not_get_here();
        }

        // Create window
        mpData->pWindow = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_position(GTK_WINDOW(mpData->pWindow), GTK_WIN_POS_CENTER_ALWAYS);
        gtk_window_set_decorated(GTK_WINDOW(mpData->pWindow), FALSE);

        GtkWidget* pVBox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
        gtk_container_add(GTK_CONTAINER(mpData->pWindow), pVBox);

        // Create label for messages
        std::string initialMsg = (list.size() > 0) ? list[0] : "Loading...";
        mpData->pLabel = gtk_label_new(initialMsg.c_str());
        gtk_label_set_justify(GTK_LABEL(mpData->pLabel), GTK_JUSTIFY_CENTER);
        gtk_label_set_lines(GTK_LABEL(mpData->pLabel), 1);
        gtk_box_pack_start(GTK_BOX(pVBox), mpData->pLabel, TRUE, FALSE, 0);

        // Create and attach progress bar
        mpData->pBar = gtk_progress_bar_new();
        gtk_box_pack_start(GTK_BOX(pVBox), mpData->pBar, TRUE, FALSE, 0);

        mpData->pulseTimerId = g_timeout_add(100, progressBarPulseeCB, mpData);
        mpData->progressUpdateId = g_timeout_add(delayInMs, progressBarUpdateCB, mpData);

        gtk_widget_show_all(mpData->pWindow);
        mpData->thread = std::thread(progressBarThread, mpData);
    }
}
