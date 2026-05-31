# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

from importlib.util import find_spec

# Register Gym environments when IsaacLab is available. Keeping this optional
# lets pure PyTorch utilities, such as SMP pretraining, import the package
# outside the IsaacLab launcher.
if find_spec("isaaclab_tasks") is not None:
    from .tasks import *  # noqa: F401, F403

# Register UI extensions only inside an Omniverse runtime.
if find_spec("omni") is not None:
    from .ui_extension_example import *  # noqa: F401, F403
