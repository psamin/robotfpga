import os
import sys

# Headless rendering: EGL on Linux (cluster, CI). macOS uses MuJoCo's native offscreen context.
# Must run before mujoco is imported anywhere.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
