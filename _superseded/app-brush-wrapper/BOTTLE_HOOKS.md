# CI / distribution hooks

The GitHub Actions workflow `.github/workflows/ios.yml` builds Nimbus3D on a
macOS runner and produces an **unsigned** `.ipa` you can sideload (AltStore,
Sideloadly) or re-sign. No Apple signing identity is used in CI.

## What the workflow does

1. `Sources/SplatEngine/build-xcframework.sh` compiles the Rust/Brush core for
   `aarch64-apple-ios` + `aarch64-apple-ios-sim` and packages
   `Frameworks/NimbusSplatCore.xcframework`.
2. `xcodegen generate` produces `Nimbus3D.xcodeproj` from `project.yml`.
3. `xcodebuild ... archive` with `CODE_SIGNING_ALLOWED=NO` builds `Nimbus3D.app`.
4. The `.app` is copied into `Payload/` and hand-zipped into
   `Nimbus3D-unsigned.ipa`, uploaded as the artifact **`Nimbus3D-unsigned-ipa`**.
5. On a `v*` tag, the IPA is also attached to a GitHub Release.

## Placeholders to fill after the repo has a remote

- `downloadURL`: OWNER/REPO placeholder. Replace once the repo has a GitHub remote, e.g.
  `https://github.com/OWNER/REPO/releases/latest/download/Nimbus3D-unsigned.ipa`.

## Prerequisites the build genuinely needs

- The Rust step must succeed. It compiles against the Brush engine
  (`brush-process`, git dep); see `Sources/SplatEngine`. Until the FFI seams in
  `brush_glue.rs` compile against a real Brush revision, the Rust step (and thus
  the whole build) stays red. That is honest: there is no offline stub for the
  trainer core by design.
- Repo-root assumption: GitHub Actions only reads `.github/workflows` at the
  repository root. This file assumes the app directory (the folder containing
  `project.yml`) **is** the repo root. If the app is nested, move the workflow to
  `<repo-root>/.github/workflows/` and add a `working-directory:` pointing at the
  app folder for the build steps.
