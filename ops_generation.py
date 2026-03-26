from __future__ import annotations

import traceback
from time import monotonic, perf_counter

import bpy
from bpy.props import BoolProperty
from bpy.types import Context, Object, Operator

from .dependency_manager import get_dependency_status
from .preferences import get_preferences
from .properties import update_source_frames_from_iterable
from .retarget import (
    analyze_prompt_segments,
    apply_static_source_motion,
    build_constraint_request,
    iter_apply_generated_motion,
)
from .runtime import (
    begin_collect_generation_job_result,
    cancel_generation_job,
    collect_generation_job_result,
    complete_generation_job,
    fail_generation_job,
    generation_job_is_active,
    generation_job_ready_to_collect,
    generation_job_result_is_loaded,
    generation_job_timed_out,
    get_generation_job_state,
    start_generation_job,
    update_generation_job_state,
)
from .runtime_setup import get_runtime_setup_status
from .utils import selected_keyframes_for_object, wrap_text_to_panel

_UI_STATE_KEYS = {
    "active": "_beyond_motion_generation_active",
    "progress": "_beyond_motion_generation_progress",
    "status_text": "_beyond_motion_generation_status_text",
    "detail_text": "_beyond_motion_generation_detail_text",
    "error_text": "_beyond_motion_generation_error_text",
}


def _active_armature_object(context: Context) -> Object | None:
    obj = context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    return None


def _current_generation_ui_state(context: Context) -> dict[str, object]:
    wm = getattr(context, "window_manager", None)
    if wm is not None and any(key in wm for key in _UI_STATE_KEYS.values()):
        return {
            "active": bool(wm.get(_UI_STATE_KEYS["active"], False)),
            "progress": float(wm.get(_UI_STATE_KEYS["progress"], 0.0) or 0.0),
            "status_text": str(wm.get(_UI_STATE_KEYS["status_text"], "") or ""),
            "detail_text": str(wm.get(_UI_STATE_KEYS["detail_text"], "") or ""),
            "error_text": str(wm.get(_UI_STATE_KEYS["error_text"], "") or ""),
        }
    return get_generation_job_state()


def _sync_progress_ui_state(context: Context, state: dict[str, object] | None = None) -> bool:
    wm = getattr(context, "window_manager", None)
    if wm is None:
        return False
    source = dict(state or get_generation_job_state())
    normalized = {
        _UI_STATE_KEYS["active"]: bool(source.get("active", False)),
        _UI_STATE_KEYS["progress"]: float(source.get("progress", 0.0) or 0.0),
        _UI_STATE_KEYS["status_text"]: str(source.get("status_text", "") or ""),
        _UI_STATE_KEYS["detail_text"]: str(source.get("detail_text", "") or ""),
        _UI_STATE_KEYS["error_text"]: str(source.get("error_text", "") or ""),
    }
    changed = False
    for key, value in normalized.items():
        if wm.get(key) != value:
            wm[key] = value
            changed = True
    return changed


def _force_progress_ui_refresh(context: Context) -> None:
    try:
        if context.area is not None:
            context.area.tag_redraw()
    except Exception:
        pass
    try:
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    except Exception:
        pass


def _runtime_ready_issue(context: Context, armature_object: Object):
    prefs = get_preferences(context)
    dependency_status = get_dependency_status(prefs.torch_device if prefs else "auto")
    if not dependency_status.ready:
        return "Install generation dependencies in Beyond Motion preferences first."
    settings = armature_object.data.beyond_motion
    runtime_status = get_runtime_setup_status(
        model_name=settings.model_name,
        text_encoder_mode=prefs.text_encoder_mode if prefs else "auto",
        text_encoder_url=prefs.text_encoder_url if prefs else "",
        checkpoint_dir_override=prefs.checkpoint_dir if prefs else "",
        hf_token=prefs.hf_token if prefs else "",
        offline_only=bool(prefs.offline_only) if prefs else False,
    )
    if not runtime_status.ready:
        return runtime_status.issues[0] if runtime_status.issues else "Finish Generation Setup first."
    return None


def _draw_prompt_preview(layout, context: Context, prompt: str) -> None:
    preview_box = layout.box()
    preview_box.label(text="Prompt Preview", icon="TEXT")
    prompt_text = (prompt or "").strip()
    if not prompt_text:
        preview_box.label(text="Enter a movement prompt to preview it here.", icon="INFO")
        return
    wrapped = wrap_text_to_panel(prompt_text, context, full_width=True, preferred_chars=42)
    for line in (wrapped.splitlines() or [""]):
        preview_box.label(text=line)


def _request_total_frame_count(request: dict[str, object] | None) -> int:
    if not request:
        return 0
    num_frames = request.get("num_frames", 0)
    if isinstance(num_frames, list):
        return sum(int(value or 0) for value in num_frames)
    return int(num_frames or 0)


def _populate_prompt_segments(settings, analyses, source_frames: list[int]) -> None:
    settings.prompt_segments.clear()
    for analysis in analyses:
        item = settings.prompt_segments.add()
        item.start_frame = int(analysis.start_frame)
        item.end_frame = int(analysis.end_frame)
        item.duration_frames = int(analysis.duration_frames)
        item.segment_kind = str(analysis.segment_kind)
        item.prompt = str(analysis.prompt)
        item.displacement = float(analysis.displacement)
        item.average_speed = float(analysis.average_speed)
        item.turn_degrees = float(analysis.turn_degrees)
    settings.prompt_segments_index = 0
    update_source_frames_from_iterable(settings, source_frames)


def _ensure_prompt_segments(context: Context, armature_object: Object) -> bool:
    settings = armature_object.data.beyond_motion
    source_frames = selected_keyframes_for_object(armature_object)
    if len(source_frames) < 2:
        return False
    expected_count = settings.expected_prompt_segment_count(source_frames)
    if settings.prompt_segments_match_frames(source_frames) and len(settings.prompt_segments) == expected_count:
        return True
    analyses = analyze_prompt_segments(context, armature_object, settings, source_frames)
    _populate_prompt_segments(settings, analyses, source_frames)
    return True


def _active_segment_prompt(settings) -> str:
    active_segment = settings.active_prompt_segment()
    if active_segment is None:
        return ""
    return str(active_segment.prompt or "")


def _draw_segment_prompt_list(layout, context: Context, armature_object: Object, *, enabled: bool) -> None:
    settings = armature_object.data.beyond_motion
    selected_frames = selected_keyframes_for_object(armature_object)

    segment_box = layout.box()
    segment_box.enabled = enabled
    segment_header = segment_box.row()
    segment_header.alert = True
    segment_header.label(text="Segment Prompts", icon="TEXT")

    process_row = segment_box.row()
    process_row.scale_y = 1.2
    process_row.operator("beyond_motion.process_keyframes", text="Process Keyframes", icon="ACTION")

    if len(selected_frames) < 2:
        segment_box.label(text="Select at least two keyframes to create prompt segments.", icon="INFO")
        return

    if not settings.prompt_segments_match_frames(selected_frames) or not settings.prompt_segments:
        segment_box.label(text="Process the selected keyframes to build one prompt per span.", icon="INFO")
        return

    list_row = segment_box.row()
    list_row.template_list(
        "BEYONDMOTION_UL_prompt_segments",
        "",
        settings,
        "prompt_segments",
        settings,
        "prompt_segments_index",
        rows=4,
    )

    active_segment = settings.active_prompt_segment()
    if active_segment is None:
        return

    detail_box = segment_box.box()
    detail_box.prop(active_segment, "segment_kind", text="Action")
    detail_box.prop(active_segment, "prompt", text="")
    detail_box.label(
        text=(
            f"{active_segment.duration_frames} frames  |  "
            f"{active_segment.displacement:.2f}m  |  "
            f"{active_segment.average_speed:.2f}m/s  |  "
            f"{active_segment.turn_degrees:+.0f} deg"
        ),
        icon="INFO",
    )
    _draw_prompt_preview(detail_box, context, active_segment.prompt)


def _draw_generation_progress(layout, context: Context) -> bool:
    state = _current_generation_ui_state(context)
    active = bool(state.get("active", False))
    status_text = str(state.get("status_text", "") or "")
    error_text = str(state.get("error_text", "") or "")
    if not active and not status_text and not error_text:
        return False

    progress_box = layout.box()
    progress_box.alert = bool(error_text)
    progress_box.label(
        text="Generation Progress" if active else "Generation Status",
        icon="TIME" if active else ("ERROR" if error_text else "CHECKMARK"),
    )
    if active and hasattr(progress_box, "progress"):
        factor = max(0.0, min(1.0, float(state.get("progress", 0.0) or 0.0)))
        progress_box.progress(factor=factor, text=f"{int(round(factor * 100.0))}%")
    detail_text = str(state.get("detail_text", "") or "")
    if status_text:
        wrapped_status = wrap_text_to_panel(status_text, context, full_width=True, preferred_chars=42)
        for index, line in enumerate(wrapped_status.splitlines() or [""]):
            progress_box.label(text=line, icon=("INFO" if not error_text else "ERROR") if index == 0 else "BLANK1")
    if detail_text and detail_text != status_text:
        wrapped_detail = wrap_text_to_panel(detail_text, context, full_width=True, preferred_chars=42)
        for line in (wrapped_detail.splitlines() or [""]):
            progress_box.label(text=line)
    if active:
        progress_box.label(text="Generation is running in the background.", icon="BLANK1")
    return True


class BEYONDMOTION_OT_generate_inbetweens(Operator):
    bl_idname = "beyond_motion.generate_inbetweens"
    bl_label = "AI Interpolation"
    bl_description = "Generate a constrained local motion segment with Kimodo and apply it back to the mapped rig"
    bl_options = {"REGISTER", "UNDO"}
    show_generation_settings: BoolProperty(default=False, options={"SKIP_SAVE"})  # type: ignore[valid-type]

    def invoke(self, context: Context, event) -> set[str]:
        del event
        armature_object = _active_armature_object(context)
        if armature_object is None:
            self.report({"ERROR"}, "Select an armature first.")
            return {"CANCELLED"}
        if generation_job_is_active():
            return context.window_manager.invoke_props_dialog(self, width=280)
        runtime_issue = _runtime_ready_issue(context, armature_object)
        if runtime_issue:
            self.report({"ERROR"}, runtime_issue)
            return {"CANCELLED"}
        source_frames = selected_keyframes_for_object(armature_object)
        if len(source_frames) < 2:
            self.report({"ERROR"}, "Select at least two keyframes in the Timeline, Dope Sheet, or Graph Editor.")
            return {"CANCELLED"}
        _ensure_prompt_segments(context, armature_object)
        return context.window_manager.invoke_props_dialog(self, width=280)

    def draw(self, context: Context) -> None:
        layout = self.layout
        armature_object = _active_armature_object(context)
        if armature_object is None:
            layout.label(text="Select an armature first.", icon="ERROR")
            return
        settings = armature_object.data.beyond_motion
        selected_frames = selected_keyframes_for_object(armature_object)
        is_running = generation_job_is_active()
        summary_box = layout.box()
        summary_box.label(text="Generation", icon="IPO_EASE_IN_OUT")
        if selected_frames:
            summary_box.label(
                text=f"Selected Keyframes: {len(selected_frames)} ({selected_frames[0]} to {selected_frames[-1]})",
                icon="ACTION",
            )
        else:
            summary_box.alert = True
            summary_box.label(text="Select at least two keyframes before generating.", icon="ERROR")

        summary_box.separator()
        summary_box.prop(settings, "root_target_mode")
        if settings.root_target_mode == "MOTION_ROOT":
            summary_box.prop_search(settings, "motion_root_bone", armature_object.data, "bones", text="Motion Root")
        summary_box.prop(settings, "blender_forward_axis")

        layout.separator()
        _draw_segment_prompt_list(layout, context, armature_object, enabled=not is_running)

        layout.separator()
        settings_box = layout.box()
        settings_box.enabled = not is_running
        header = settings_box.row(align=True)
        header.prop(
            self,
            "show_generation_settings",
            text="",
            icon="TRIA_DOWN" if self.show_generation_settings else "TRIA_RIGHT",
            emboss=False,
        )
        header.label(text="Generation Settings", icon="SETTINGS")
        if not self.show_generation_settings:
            layout.separator()
            _draw_generation_progress(layout, context)
            return
        settings_box.prop(settings, "model_name")
        settings_box.prop(settings, "diffusion_steps")
        settings_box.prop(settings, "cfg_type")
        if settings.cfg_type != "nocfg":
            settings_box.prop(settings, "cfg_text_weight")
            if settings.cfg_type == "separated":
                settings_box.prop(settings, "cfg_constraint_weight")
        settings_box.prop(settings, "seed")
        settings_box.prop(settings, "apply_postprocess")
        settings_box.prop(settings, "hold_frame_bias")
        settings_box.prop(settings, "keypose_match_frames")
        settings_box.prop(settings, "use_locomotion_root_path")
        layout.separator()
        _draw_generation_progress(layout, context)

    def _tag_relevant_redraw(self) -> None:
        window_manager = bpy.context.window_manager
        for window in window_manager.windows:
            screen = window.screen
            if screen is None:
                continue
            for area in screen.areas:
                if area.type in {"VIEW_3D", "DOPESHEET_EDITOR", "GRAPH_EDITOR"}:
                    area.tag_redraw()

    def _finish_modal(self, context: Context) -> None:
        _sync_progress_ui_state(context)
        timer = getattr(self, "_timer", None)
        if timer is not None:
            try:
                context.window_manager.event_timer_remove(timer)
            except Exception:
                pass
            self._timer = None
        try:
            context.window_manager.progress_end()
        except Exception:
            pass
        self._tag_relevant_redraw()
        _force_progress_ui_refresh(context)

    def _report_progress_if_needed(self) -> None:
        state = get_generation_job_state()
        if not bool(state.get("active", False)):
            return
        phase = str(state.get("phase", "") or "")
        progress = max(0.0, min(1.0, float(state.get("progress", 0.0) or 0.0)))
        progress_step = int(progress * 100.0) // 5
        if phase == "starting":
            report_key = (phase, -1)
        else:
            report_key = (phase, progress_step)
        previous_report_key = getattr(self, "_last_report_key", None)
        if not isinstance(previous_report_key, tuple) or len(previous_report_key) < 2:
            previous_report_key = (None, None)
        if report_key == previous_report_key:
            return
        now = monotonic()
        last_report_at = float(getattr(self, "_last_report_at", 0.0) or 0.0)
        status_text = str(state.get("status_text", "Generating in-betweens..."))
        if report_key[0] != previous_report_key[0] or (now - last_report_at) >= 0.35:
            self.report({"INFO"}, status_text)
            self._last_report_key = report_key
            self._last_report_at = now

    def _step_apply_iterator(self, context: Context) -> bool:
        apply_iterator = getattr(self, "_apply_iterator", None)
        if apply_iterator is None:
            return False

        deadline = perf_counter() + 0.004
        stepped = False
        while perf_counter() < deadline:
            try:
                step = next(apply_iterator)
            except StopIteration:
                response = getattr(self, "_pending_response", {})
                request = getattr(self, "_request", {})
                settings_model_name = str(request.get("model_name", "kimodo"))
                device = response.get("device", "unknown") if isinstance(response, dict) else "unknown"
                success_text = (
                    f"Generated {_request_total_frame_count(request)} frames with {settings_model_name} on {device}."
                )
                complete_generation_job(success_text)
                self.report({"INFO"}, success_text)
                worker_log = str(getattr(self, "_pending_worker_log", "") or "")
                if worker_log:
                    print(worker_log)
                self._apply_iterator = None
                self._finish_modal(context)
                return True
            stepped = True
            step_progress = max(0.0, min(1.0, float(step.get("progress", 0.0) or 0.0)))
            update_generation_job_state(
                active=True,
                phase="applying",
                progress=0.96 + (0.04 * step_progress),
                status_text=str(step.get("status_text", "Applying generated motion in Blender...")),
                detail_text=str(step.get("detail_text", "")),
                error_text="",
            )
            _sync_progress_ui_state(context)
            if perf_counter() >= deadline:
                break
        if stepped:
            self._report_progress_if_needed()
        return False

    def modal(self, context: Context, event) -> set[str]:
        if event.type != "TIMER":
            return {"RUNNING_MODAL"}

        self._tag_relevant_redraw()
        state = get_generation_job_state()
        if _sync_progress_ui_state(context, state):
            _force_progress_ui_refresh(context)
        try:
            context.window_manager.progress_update(max(0.0, min(100.0, float(state.get("progress", 0.0) or 0.0) * 100.0)))
        except Exception:
            pass

        if getattr(self, "_apply_iterator", None) is not None:
            try:
                finished = self._step_apply_iterator(context)
                if finished:
                    return {"FINISHED"}
                return {"RUNNING_MODAL"}
            except Exception as error:
                traceback.print_exc()
                fail_generation_job(str(error))
                self.report({"ERROR"}, str(error))
                self._finish_modal(context)
                return {"CANCELLED"}

        self._report_progress_if_needed()

        prefs = get_preferences(context)
        timeout_seconds = prefs.job_timeout_seconds if prefs is not None else 600
        if generation_job_timed_out(timeout_seconds):
            timeout_message = "Beyond Motion timed out while generating local motion."
            cancel_generation_job(timeout_message)
            self.report({"ERROR"}, timeout_message)
            self._finish_modal(context)
            return {"CANCELLED"}

        if not generation_job_ready_to_collect():
            return {"RUNNING_MODAL"}
        begin_collect_generation_job_result()
        if not generation_job_result_is_loaded():
            return {"RUNNING_MODAL"}

        try:
            output, response, worker_log = collect_generation_job_result()
            armature_name = getattr(self, "_armature_name", "")
            armature_object = bpy.data.objects.get(armature_name)
            if armature_object is None or armature_object.type != "ARMATURE":
                raise RuntimeError("The target armature is no longer available for Beyond Motion.")
            settings = armature_object.data.beyond_motion
            self._pending_response = response
            self._pending_worker_log = worker_log
            self._apply_iterator = iter_apply_generated_motion(context, armature_object, settings, self._source_data, output)
            update_generation_job_state(
                active=True,
                phase="applying",
                progress=max(float(get_generation_job_state().get("progress", 0.0) or 0.0), 0.96),
                status_text="Applying generated motion in Blender...",
                detail_text="Writing the generated motion back to the rig in small batches.",
                error_text="",
            )
            _sync_progress_ui_state(context)
            _force_progress_ui_refresh(context)
            self.report({"INFO"}, "Applying generated motion in Blender...")
            return {"RUNNING_MODAL"}
        except Exception as error:
            traceback.print_exc()
            fail_generation_job(str(error))
            self.report({"ERROR"}, str(error))
            self._finish_modal(context)
            return {"CANCELLED"}

    def execute(self, context: Context) -> set[str]:
        armature_object = _active_armature_object(context)
        if armature_object is None:
            return {"CANCELLED"}
        if generation_job_is_active():
            self.report({"INFO"}, "Beyond Motion is already generating in-betweens.")
            return {"CANCELLED"}

        settings = armature_object.data.beyond_motion
        settings.ensure_human_bones()
        runtime_issue = _runtime_ready_issue(context, armature_object)
        if runtime_issue:
            self.report({"ERROR"}, runtime_issue)
            return {"CANCELLED"}

        source_frames = selected_keyframes_for_object(armature_object)
        if len(source_frames) < 2:
            self.report({"ERROR"}, "Select at least two keyframes in the Timeline, Dope Sheet, or Graph Editor.")
            return {"CANCELLED"}
        if not _ensure_prompt_segments(context, armature_object):
            self.report({"ERROR"}, "Process the selected keyframes before generating.")
            return {"CANCELLED"}

        try:
            request, source_data = build_constraint_request(context, armature_object, settings, source_frames)
            if request is None or not source_data.generation_required:
                apply_static_source_motion(context, armature_object, settings, source_data)
                response = {"device": "none"}
                worker_log = ""
            else:
                start_generation_job(context, request)
                self._armature_name = armature_object.name
                self._source_data = source_data
                self._request = request
                self._apply_iterator = None
                self._pending_response = None
                self._pending_worker_log = ""
                self._last_report_key = None
                self._last_report_at = 0.0
                self._timer = context.window_manager.event_timer_add(0.05, window=context.window)
                try:
                    context.window_manager.progress_begin(0.0, 100.0)
                except Exception:
                    pass
                _sync_progress_ui_state(context)
                context.window_manager.modal_handler_add(self)
                self.report({"INFO"}, "Starting AI in-between generation...")
                self._tag_relevant_redraw()
                _force_progress_ui_refresh(context)
                return {"RUNNING_MODAL"}
        except Exception as error:
            traceback.print_exc()
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        if worker_log:
            print(worker_log)
        device = response.get("device", "unknown")
        if request is None or not source_data.generation_required:
            frame_count = source_frames[-1] - source_frames[0] + 1
            self.report({"INFO"}, f"Applied a static hold across {frame_count} frames.")
        else:
            self.report(
                {"INFO"},
                f"Generated {_request_total_frame_count(request)} frames with {settings.model_name} on {device}.",
            )
        return {"FINISHED"}


class BEYONDMOTION_OT_process_keyframes(Operator):
    bl_idname = "beyond_motion.process_keyframes"
    bl_label = "Process Keyframes"
    bl_description = "Analyze the selected keyframes and build one editable AI prompt per keyframe span"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: Context) -> set[str]:
        armature_object = _active_armature_object(context)
        if armature_object is None:
            self.report({"ERROR"}, "Select an armature first.")
            return {"CANCELLED"}
        source_frames = selected_keyframes_for_object(armature_object)
        if len(source_frames) < 2:
            self.report({"ERROR"}, "Select at least two keyframes to process.")
            return {"CANCELLED"}

        settings = armature_object.data.beyond_motion
        try:
            analyses = analyze_prompt_segments(context, armature_object, settings, source_frames)
            _populate_prompt_segments(settings, analyses, source_frames)
        except Exception as error:
            traceback.print_exc()
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        self.report({"INFO"}, f"Processed {len(analyses)} keyframe spans.")
        return {"FINISHED"}
