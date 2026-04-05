# Release Notes

This page tracks user-visible enhancements and behavior changes introduced in this repository.

## 2026-04-05

### WebUI usability and localization

- Added per-field help `?` popups in LlamaBoard for interactive controls.
- Added a Zena avatar quick menu for language and theme toggles.
- Expanded language options in WebUI to: `en`, `ro`, `hu`, `he`, `fr`, `de`, `es`, `pt`.
- Added language fallback to English when a locale key is missing.
- Improved avatar image fallback order to avoid missing-header-image states.

### Export workflow UX

- Improved GGUF export ergonomics with clearer quantization guidance and safer defaults.
- Added clearer advanced export options and explanatory helper text.

### Operations and verification notes

- Added client-side wiring that attaches help controls to dynamically rendered Gradio components.
- Reworked UI script bootstrap path so help/menu scripts are loaded reliably during page startup.

## How to update this file

When you merge user-visible changes:

1. Add a new date section at the top.
2. Group updates by area (for example: WebUI, Training, Inference, Export).
3. Keep each bullet concise and behavior-focused.
