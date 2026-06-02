# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

from importlib.util import find_spec

# EN: Register Gym environments only inside an Isaac Sim runtime. Pure PyTorch
# tools such as SMP pretraining/sampling must not import IsaacLab, USD, or Kit.
# 中文：只有在 Isaac Sim 运行时才注册 Gym 环境。离线的 SMP 预训练/采样工具
# 不应触发 IsaacLab、USD 或 Kit 的导入，否则普通 python 里会缺 pxr。
if find_spec("isaaclab_tasks") is not None and find_spec("pxr") is not None:
    from .tasks import *  # noqa: F401, F403

# EN: UI extensions are available only when Omniverse modules are loaded.
# 中文：UI 扩展只在 Omniverse 模块存在时注册。
if find_spec("omni") is not None:
    from .ui_extension_example import *  # noqa: F401, F403
