"""
This is a simple example of using Falcor for a compute shader simulation.
"""

import falcor
from pathlib import Path
import time
import numpy as np

DIR = Path(__file__).parent

BALL_COUNT = 100
BALL_RADIUS = 0.1
RESOLUTION = 1024

# create testbed instance with a window
testbed = falcor.Testbed(create_window=True, width=1024, height=1024)
device = testbed.device

# create a structured buffer to hold our ball postitions (float2) and velocities (float2)
balls = device.create_structured_buffer(
    struct_size=16,
    element_count=BALL_COUNT,
    bind_flags=falcor.ResourceBindFlags.ShaderResource
    | falcor.ResourceBindFlags.UnorderedAccess,
)

# create a texture to render to
texture = device.create_texture(
    format=falcor.ResourceFormat.RGBA32Float,
    width=RESOLUTION,
    height=RESOLUTION,
    mip_levels=1,
    bind_flags=falcor.ResourceBindFlags.UnorderedAccess
    | falcor.ResourceBindFlags.ShaderResource,
)

# assign the texture to be rendered in the testbed window
testbed.render_texture = texture

# disable the testbed UI
testbed.show_ui = False

# create a compute pass for updating balls
update_balls = falcor.ComputePass(
    device, file=DIR / "balls_update.cs.slang", cs_entry="main"
)
update_balls.globals.g_balls = balls
update_balls.globals.g_ball_count = BALL_COUNT

# create a compute pass for rendering balls
render_balls = falcor.ComputePass(
    device, file=DIR / "balls_render.cs.slang", cs_entry="main"
)
render_balls.globals.g_balls = balls
render_balls.globals.g_ball_count = BALL_COUNT
render_balls.globals.g_ball_radius = BALL_RADIUS
render_balls.globals.g_output = texture
render_balls.globals.g_resolution = RESOLUTION

# initialize balls with random position and velocities
balls.from_numpy(np.random.rand(BALL_COUNT, 4).astype(np.float32) * 2 - 1)

prev_time = time.time()

profiler = device.profiler

# main loop
while not testbed.should_close:
    # compute seconds since last frame
    cur_time = time.time()
    dt = cur_time - prev_time
    prev_time = cur_time

    # update and render the balls
    update_balls.globals.g_dt = dt
    with profiler.event("update_balls"):
        update_balls.execute(threads_x=BALL_COUNT)
    with profiler.event("render_balls"):
        render_balls.execute(threads_x=RESOLUTION, threads_y=RESOLUTION)

    # present the rendered texture
    testbed.frame()
