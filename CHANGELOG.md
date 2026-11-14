# Changelog

All notable changes to `waterui-image` are documented in this file.

## [Unreleased]

## [0.6.0](https://github.com/water-rs/image/compare/v0.5.0...v0.6.0) - 2026-11-14

### Added

- [**breaking**] remove the registration() API that status() replaced
- keep registration() beside status() for the media transition
- [**breaking**] draw through the waterui Canvas API
- [**breaking**] implement the response-geometry layout query
- *(scene)* record through the render-target-neutral draw contract ([#41](https://github.com/water-rs/image/pull/41))
- [**breaking**] port to WaterUI's final scene-resource API

### Fixed

- build against the framework's current wgpu and shaderloom
- *(image)* report a refused image registration instead of panicking
- *(image)* scale non-resizable images down to fit proposals
- *(deps)* resolve gpu-allocator against windows 0.62
- *(scene)* implement SceneContent::rebuild_for_engine for the 1406 contract
- *(lint)* satisfy clippy 1.99 assert_is_empty
- repin hydrolysis to 14d5e231, m3 to fcb8b9e8; patch all registry waterkit crates to 123b0581
- pin hydrolysis-m3 to migrated rev; import Signal in image tests
- *(lock)* bind gpu-allocator to the windows version wgpu-hal's DX12 backend uses
- *(image)* decode still images without the codec default features
- *(ci)* pin the toolchain without disarming the MSRV leg ([#30](https://github.com/water-rs/image/pull/30))

### Other

- *(deps)* point migrated Batch 2 git dependencies to main ([#71](https://github.com/water-rs/image/pull/71))
- update release-plz rev for Rust 1.100 compatibility
- release from main and retire dev ([#66](https://github.com/water-rs/image/pull/66))
- restore the Hydrolysis test fonts from the Actions cache ([#65](https://github.com/water-rs/image/pull/65))
- *(deps)* point migrated Batch 1 git dependencies to main ([#62](https://github.com/water-rs/image/pull/62)) ([#63](https://github.com/water-rs/image/pull/63))
- compare rendered colours by CIEDE2000
- name the offscreen renderer through waterui_graphics::gpu
- follow the split repositories' dev again
- an empty reactive image has no accessibility element
- authenticated font downloads and cached dxc
- require the framework's Rust 1.99
- follow the framework's dev for response geometry
- Report a settled registration state instead of failures only
- Merge pull request #52 from water-rs/fix/42-registration-error
- pass the release pull request's gate without CI
- *(deps)* pin water-rs/waterui to dev at e76ad1e
- Merge branch 'dev' into chore/1924-surface-wake
- *(deps)* repin water-rs/waterui to the surface-wake head
- install Hydrolysis test fonts before tests
- *(deps)* repin water-rs/waterui to 1e359bca
- *(scene)* record through Content::record on the in-tree recorder API
- *(deps)* repin the engine and framework onto in-tree Cherenkov
- *(deps)* pin hydrolysis, hydrolysis-m3 and video-gpu at the repinned heads
- pin cherenkov to 52236561
- install SPIRV-Tools through the shared Linux deps action
- install SPIRV-Tools for cherenkov-gpu's shader build on non-Apple legs
- run the mounted checks unconditionally
- repin to the Cherenkov cutover, restore host-e2e deps
- repin to waterui 8fd2a851, cherenkov 8d00976e, waterkit 2f1e359
- repin cherenkov to its rewritten history
- [**breaking**] port scene recording to cherenkov content
- request waterkit-codec 'software-decode'
- link the AVIF software decoder only where no platform decoder exists
- repin the vello fork to the glifo canonical-location fix
- repin water-rs dependencies to merged dev heads
- repin water-rs dependencies to unified migration heads
- repin water-rs dependencies to unified migration heads
- repin hydrolysis to accessibility import fix
- refresh lockfile for final branch revs
- repin migrated dependency branch heads
- [**breaking**] migrate to nami snapshot API
- disable incremental builds and trim debuginfo ([#16](https://github.com/water-rs/image/pull/16))
- consume the 0.5.0 framework wave from crates.io ([#29](https://github.com/water-rs/image/pull/29))

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
