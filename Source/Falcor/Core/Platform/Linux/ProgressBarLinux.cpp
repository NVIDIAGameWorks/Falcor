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
#include "Core/Platform/ProgressBar.h"
#include "Core/Error.h"

#include <gtk/gtk.h>

#include <chrono>
#include <thread>

namespace Falcor
{
struct ProgressBar::Window
{
    bool running;
    std::thread thread;

    Window(const std::string& msg)
    {
        running = true;
        thread = std::thread(threadFunc, this, msg);
    }

    ~Window()
    {
        running = false;
        thread.join();
    }

    static void threadFunc(ProgressBar::Window* pThis, std::string msg)
    {
        // Create window
        GtkWidget* pWindow = gtk_window_new(GTK_WINDOW_TOPLEVEL);
        gtk_window_set_position(GTK_WINDOW(pWindow), GTK_WIN_POS_CENTER_ALWAYS);
        gtk_window_set_decorated(GTK_WINDOW(pWindow), FALSE);

        GtkWidget* pVBox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
        gtk_container_add(GTK_CONTAINER(pWindow), pVBox);

        // Create label for message
        GtkWidget* pLabel = gtk_label_new(msg.c_str());
        gtk_label_set_justify(GTK_LABEL(pLabel), GTK_JUSTIFY_CENTER);
        gtk_label_set_lines(GTK_LABEL(pLabel), 1);
        gtk_box_pack_start(GTK_BOX(pVBox), pLabel, TRUE, FALSE, 0);

        // Create and attach progress bar
        GtkWidget* pBar = gtk_progress_bar_new();
        gtk_box_pack_start(GTK_BOX(pVBox), pBar, TRUE, FALSE, 0);

        guint pulseTimerId = g_timeout_add(100, pulseTimer, pBar);

        gtk_widget_show_all(pWindow);

        while (pThis->running || gtk_events_pending())
        {
            gtk_main_iteration_do(FALSE);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        gtk_window_close(GTK_WINDOW(pWindow));

        g_source_remove(pulseTimerId);
        gtk_widget_destroy(pWindow);
    }

    // At regular intervals, pulse the progress bar
    static gboolean pulseTimer(gpointer pGtkData)
    {
        GtkWidget* pBar = (GtkWidget*)pGtkData;
        gtk_progress_bar_pulse(GTK_PROGRESS_BAR(pBar));
        return TRUE;
    }
};

ProgressBar::ProgressBar() {}

ProgressBar::~ProgressBar()
{
    close();
}

void ProgressBar::show(const std::string& msg)
{
    close();

    if (!gtk_init_check(0, nullptr))
        FALCOR_THROW("Failed to initialize GTK.");

    mpWindow = std::make_unique<Window>(msg);
}

void ProgressBar::close()
{
    mpWindow.reset();
}
} // namespace Falcor
