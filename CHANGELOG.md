# Changelog

All notable changes to `waterui-image` are documented in this file.

## [Unreleased]

## [0.4.0](https://github.com/water-rs/image/compare/v0.3.0...v0.4.0) - 2026-09-11

### Added

- give the image scene contents an intrinsic size
- [**breaking**] draw images through Scene2D instead of a wgpu pipeline

### Fixed

- *(ci)* install the same Linux packages for the release preflight
- *(release)* close package rehearsal gaps
- *(release)* verify registry-only package graph
- fix repository rule violations and refresh documentation

### Other

- link the test graph to the waterui 0.4 release commit
- *(deps)* waterui-graphics 0.4 (and waterui-text/-testing 0.4 where used)
- refresh git dependency revisions after the upstream history rewrite
- update Linux package matrix and add dxc on Windows
- setup standalone crate files, CI workflows, and release-plz
- *(deps)* drop `image`'s AVIF encoder from every consumer
- consolidate GPU glue into waterui-graphics helpers
- ship the licence texts in every published crate
- depend on shaderloom directly, and give the icon codegen its own name
- prepare WaterUI 0.3 release versions
- Fix workspace CI failures
- Make reactivity precise across renderers
- upgrade workspace dependencies
- Add cross-platform shader AOT with Shaderloom
- refactor native backends and GPU surface integration
- *(cli)* template the preview perf report with askama
- achieve zero clippy warnings across the workspace
- clean up clippy warnings across the workspace
- SubView: Send + Sync; decouple GpuView from SubView
- Lean dependency graph for embedded: gpu/widgets/gestures features
- Restore WaterUI CI gates and reactive map API
- reorganize the project

## [0.3.0](https://github.com/water-rs/waterui/compare/image-v0.2.1...image-v0.3.0) - 2026-08-25

- Added GPU-first image realization, filter integration, color handling, and WaterKit codec support.
