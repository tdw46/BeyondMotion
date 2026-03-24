# Kimodo Integration Notes

These notes summarize the parts of `vendor/kimodo` that directly shaped the Blender extension design.

## Relevant Kimodo docs

- `vendor/kimodo/README.md`
- `vendor/kimodo/docs/source/getting_started/installation.md`
- `vendor/kimodo/docs/source/getting_started/quick_start.md`
- `vendor/kimodo/docs/source/key_concepts/constraints.md`
- `vendor/kimodo/docs/source/user_guide/cli.md`
- `vendor/kimodo/docs/source/user_guide/constraints.md`
- `vendor/kimodo/docs/source/key_concepts/limitations.md`
- `vendor/kimodo/docs/source/user_guide/output_formats.md`

## Key findings

- Human motion generation should target the SOMA models first. They are the default and best-supported human option in the docs.
- Kimodo already supports sparse full-body keyframe constraints at arbitrary frame indices, which maps cleanly to Blender inbetween generation.
- Constraint JSON can be authored directly with:
  - `type = "fullbody"`
  - `frame_indices`
  - `local_joints_rot`
  - `root_positions`
  - optional `smooth_root_2d`
- For SOMA models, 77-joint constraint files are accepted and converted to the model's internal 30-joint representation when needed.
- Multi-prompt generation exists, but this first Blender pass uses a single prompt across one constrained segment.

## Architecture decisions taken here

- Run Kimodo in a separate local Python process from Blender.
- Use the vendored repo in `vendor/kimodo` as the worker's code source.
- Keep Blender responsible for:
  - humanoid bone mapping
  - source frame capture
  - constraint export
  - generated keyframe application
- Use Beyond Motion's internal humanoid auto-detection so the extension stays independent of any sibling add-ons.

## Apple Silicon / MPS notes

- The worker now accepts a requested torch device and resolves `auto` as CUDA, then MPS, then CPU.
- When MPS is enabled, the worker enables PyTorch's CPU fallback path by default.
- Bundled local text encoding now defaults to an open 8B Hugging Face embedding model instead of the original gated Meta-Llama LLM2Vec path.
- Kimodo's local text encoder dtype is still forced off `bfloat16` on MPS and uses `float16` unless `TEXT_ENCODER_DTYPE` overrides it.
- The vendored Kimodo CLI and local text-encoder runtime were patched so an explicit `mps` device can propagate all the way into text encoding instead of silently dropping back to CPU.

## Extension-local dependency flow

- The add-on now installs generation dependencies into `_vendor/` inside the extension itself by calling Blender's bundled Python with `pip install --target`.
- Download caches and install logs live under `wheels/`.
- The main Beyond panel stays locked until the dependency bootstrap reports a compatible install for the selected runtime device.

## Current assumptions and limitations

- Retargeting is semantic and local-rotation based.
- It works best on FK rigs whose local bone axes are reasonably conventional.
- Root translation is applied as deltas from Kimodo's generated hips trajectory onto a chosen Blender target:
  - hips bone
  - separate motion root bone
  - armature object
- This pass does not yet expose end-effector-only constraints, dense root paths, or multi-prompt timeline authoring.
