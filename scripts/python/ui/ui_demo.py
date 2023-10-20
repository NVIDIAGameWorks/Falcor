"""
This is a simple example of using UI elements in Python.
"""

import falcor
from falcor import ui
import threading


class WidgetWindow:
    """
    Demonstrates all the widgets currently available in Falcor.
    """

    def __init__(self, screen: ui.Screen):
        self.window = window = ui.Window(
            parent=screen, title="Widgets", position=[900, 10], size=[500, 1000]
        )
        self.widgets = []

        basic_group = ui.Group(parent=window, label="Basic widgets")

        text = ui.Text(parent=basic_group, text="Text widget\nwith multiple lines\nof plain text")

        button = ui.Button(
            parent=basic_group,
            label="Button widget",
            callback=lambda: print("Button clicked!"),
        )

        checkbox = ui.Checkbox(
            parent=basic_group,
            label="Checkbox widget",
            value=True,
            change_callback=lambda: print(
                f"Checkbox value changed to {checkbox.value}"
            ),
        )

        combobox = ui.Combobox(
            parent=basic_group,
            label="Combobox widget",
            items=["Item 1", "Item 2", "Item 3"],
            value=1,
            change_callback=lambda: print(
                f"Combobox value changed to {combobox.value} ({combobox.items[combobox.value]})"
            ),
        )

        progress_bar = ui.ProgressBar(parent=basic_group, fraction=0.5)

        self.widgets += [text, button, checkbox, combobox, progress_bar]

        drag_group = ui.Group(parent=window, label="Drag widgets")

        drag_int = ui.DragInt(
            parent=drag_group,
            label="DragInt widget",
            value=10,
            change_callback=lambda: print(f"DragInt value changed to {drag_int.value}"),
        )

        drag_int2 = ui.DragInt2(
            parent=drag_group,
            label="DragInt2 widget",
            value=[10, 20],
            change_callback=lambda: print(
                f"DragInt2 value changed to {drag_int2.value}"
            ),
        )

        drag_int3 = ui.DragInt3(
            parent=drag_group,
            label="DragInt3 widget",
            value=[10, 20, 30],
            change_callback=lambda: print(
                f"DragInt3 value changed to {drag_int3.value}"
            ),
        )

        drag_int4 = ui.DragInt4(
            parent=drag_group,
            label="DragInt4 widget",
            value=[10, 20, 30, 40],
            change_callback=lambda: print(
                f"DragInt4 value changed to {drag_int4.value}"
            ),
        )

        drag_float = ui.DragFloat(
            parent=drag_group,
            label="DragFloat widget",
            value=0.5,
            speed=0.1,
            change_callback=lambda: print(
                f"DragFloat value changed to {drag_float.value}"
            ),
        )

        drag_float2 = ui.DragFloat2(
            parent=drag_group,
            label="DragFloat2 widget",
            value=[0.5, 0.6],
            speed=0.1,
            change_callback=lambda: print(
                f"DragFloat2 value changed to {drag_float2.value}"
            ),
        )

        drag_float3 = ui.DragFloat3(
            parent=drag_group,
            label="DragFloat3 widget",
            value=[0.5, 0.6, 0.7],
            speed=0.1,
            change_callback=lambda: print(
                f"DragFloat3 value changed to {drag_float3.value}"
            ),
        )

        drag_float4 = ui.DragFloat4(
            parent=drag_group,
            label="DragFloat4 widget",
            value=[0.5, 0.6, 0.7, 0.8],
            speed=0.1,
            change_callback=lambda: print(
                f"DragFloat4 value changed to {drag_float4.value}"
            ),
        )

        self.widgets += [
            drag_int,
            drag_int2,
            drag_int3,
            drag_int4,
            drag_float,
            drag_float2,
            drag_float3,
            drag_float4,
        ]

        slider_group = ui.Group(parent=window, label="Slider widgets")

        slider_int = ui.SliderInt(
            parent=slider_group,
            label="SliderInt widget",
            value=0,
            min=-100,
            max=100,
            change_callback=lambda: print(
                f"SliderInt value changed to {slider_int.value}"
            ),
        )

        slider_int2 = ui.SliderInt2(
            parent=slider_group,
            label="SliderInt2 widget",
            value=falcor.int2(0),
            min=-100,
            max=100,
            change_callback=lambda: print(
                f"SliderInt2 value changed to {slider_int2.value}"
            ),
        )

        slider_int3 = ui.SliderInt3(
            parent=slider_group,
            label="SliderInt3 widget",
            value=falcor.int3(0),
            min=-100,
            max=100,
            change_callback=lambda: print(
                f"SliderInt3 value changed to {slider_int3.value}"
            ),
        )

        slider_int4 = ui.SliderInt4(
            parent=slider_group,
            label="SliderInt4 widget",
            value=falcor.int4(0),
            min=-100,
            max=100,
            change_callback=lambda: print(
                f"SliderInt4 value changed to {slider_int4.value}"
            ),
        )

        slider_float = ui.SliderFloat(
            parent=slider_group,
            label="SliderFloat widget",
            value=0,
            min=-100,
            max=100,
            change_callback=lambda: print(
                f"SliderFloat value changed to {slider_float.value}"
            ),
        )

        slider_float2 = ui.SliderFloat2(
            parent=slider_group,
            label="SliderFloat2 widget",
            value=falcor.float2(0),
            min=-100,
            max=100,
            change_callback=lambda: print(
                f"SliderFloat2 value changed to {slider_float2.value}"
            ),
        )

        slider_float3 = ui.SliderFloat3(
            parent=slider_group,
            label="SliderFloat3 widget",
            value=falcor.float3(0),
            min=-100,
            max=100,
            change_callback=lambda: print(
                f"SliderFloat3 value changed to {slider_float3.value}"
            ),
        )

        slider_float4 = ui.SliderFloat4(
            parent=slider_group,
            label="SliderFloat4 widget",
            value=falcor.float4(0),
            min=-100,
            max=100,
            change_callback=lambda: print(
                f"SliderFloat4 value changed to {slider_float4.value}"
            ),
        )

        self.widgets += [
            slider_int,
            slider_int2,
            slider_int3,
            slider_int4,
            slider_float,
            slider_float2,
            slider_float3,
            slider_float4,
        ]

        empty_group = ui.Group(parent=window, label="Empty group")

        self.widgets += [empty_group]

        ui.Button(
            parent=window,
            label="Disable all widgets",
            callback=lambda: self.set_widgets_enabled(False),
        )
        ui.Button(
            parent=window,
            label="Enable all widgets",
            callback=lambda: self.set_widgets_enabled(True),
        )

    def set_widgets_enabled(self, enabled: bool):
        for widget in self.widgets:
            widget.enabled = enabled


class DemoWindow:
    def __init__(self, screen: ui.Screen, widget_window: WidgetWindow):
        self.widget_window = widget_window
        self.window = window = ui.Window(
            parent=screen, title="Demo Window", position=[300, 10], size=[500, 1000]
        )

        ui.Text(parent=window, text="\nControl the widget window from here:")

        # Buttons to show/close widget window
        ui.Button(
            parent=window,
            label="Show widget window",
            callback=lambda: self.widget_window.window.show(),
        )
        ui.Button(
            parent=window,
            label="Close widget window",
            callback=lambda: self.widget_window.window.close(),
        )

        ui.Text(parent=window, text="\nControl the position/size of this window:")

        # Create button to move this window
        def move_window():
            self.window.position = [50, 50]

        ui.Button(parent=window, label="Move window to [50, 50]", callback=move_window)

        # Create button to resize this window, specify callback after creating button
        def resize_window():
            self.window.size = [500, 500]

        button = ui.Button(parent=window, label="Resize window to [500, 500]")
        button.callback = resize_window

        ui.Text(parent=window, text="\nA simple counter button:")

        # Simple counter button
        self.counter = 0
        self.counter_button = ui.Button(
            parent=window, label="Clicked 0 times", callback=self.count
        )

        ui.Text(
            parent=window, text="\nA simple button to start/stop some background task:"
        )

        # Start/stop button that changes label
        self.running = False
        self.timer = None
        self.start_stop_button = ui.Button(
            parent=window, label="Start", callback=self.start_stop
        )

        # Progress bar
        self.progress_bar = ui.ProgressBar(parent=window, fraction=0)

    def count(self):
        self.counter += 1
        self.counter_button.label = f"Clicked {self.counter} times"

    def timer_callback(self):
        self.timer = None
        if self.running:
            self.progress_bar.fraction += 0.01
            if self.progress_bar.fraction < 1:
                self.timer = threading.Timer(0.05, self.timer_callback)
                self.timer.start()

    def start_stop(self):
        if self.running:
            self.running = False
            self.start_stop_button.label = "Start"
            self.progress_bar.fraction = 0
            if self.timer:
                self.timer.cancel()
            self.timer = None
        else:
            self.running = True
            self.start_stop_button.label = "Stop"
            self.timer = threading.Timer(0.05, self.timer_callback)
            self.timer.start()


# Create testbed and windows
testbed = falcor.Testbed(create_window=True, width=1920, height=1080)
widget_window = WidgetWindow(screen=testbed.screen)
demo_window = DemoWindow(screen=testbed.screen, widget_window=widget_window)

# We need to run frame-by-frame to have Python's timer working
while not testbed.should_close:
    testbed.frame()
