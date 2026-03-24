from __future__ import annotations

import importlib
import sys

from . import dependency_manager
from . import auto_load


dependency_manager.ensure_runtime_paths()


def register() -> None:
    auto_load.register()

    props_mod = auto_load.get_module("properties")
    if props_mod and hasattr(props_mod, "register_properties"):
        props_mod.register_properties()

    ui_mod = auto_load.get_module("ui_panels")
    if ui_mod and hasattr(ui_mod, "register_header_draw"):
        ui_mod.register_header_draw()

    runtime_setup_mod = auto_load.get_module("runtime_setup")
    if runtime_setup_mod and hasattr(runtime_setup_mod, "register_startup_auth_check"):
        runtime_setup_mod.register_startup_auth_check()


def unregister() -> None:
    ui_mod = auto_load.get_module("ui_panels")
    if ui_mod and hasattr(ui_mod, "unregister_header_draw"):
        try:
            ui_mod.unregister_header_draw()
        except Exception as error:  # pragma: no cover - Blender unregister safety
            print(f"BeyondMotion: failed to unregister header draw: {error}")

    props_mod = auto_load.get_module("properties")
    if props_mod and hasattr(props_mod, "unregister_properties"):
        try:
            props_mod.unregister_properties()
        except Exception as error:  # pragma: no cover - Blender unregister safety
            print(f"BeyondMotion: failed to unregister properties: {error}")

    auto_load.unregister()

    try:
        pkg = __package__
        to_delete = [name for name in list(sys.modules.keys()) if name == pkg or name.startswith(pkg + ".")]
        for name in to_delete:
            del sys.modules[name]
        importlib.invalidate_caches()
    except Exception as error:  # pragma: no cover - Blender unregister safety
        print(f"BeyondMotion: sys.modules purge warning: {error}")
