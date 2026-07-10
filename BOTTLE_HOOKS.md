# Bottle hooks: the exact values Bottle needs from Nimbus3D

_Companion to `BOTTLE_INTEGRATION.md` (same directory), which describes the Bottle-side
code changes. This file is the Nimbus3D-side contract: the concrete `source.json` entry
and where the IPA comes from. Hand both files to the Bottle session together._

## Where the IPA comes from

`.github/workflows/ios.yml` in this repo builds an UNSIGNED `Nimbus3D.ipa`
(macOS runner, `xcodebuild archive` with `CODE_SIGNING_ALLOWED=NO`, then the
`.app` zipped as `Payload/Nimbus3D.app`). There are no Apple certs in CI;
Bottle re-signs at install time with the user's free Apple ID.

- Every push to `main` uploads the IPA as a workflow artifact named `Nimbus3D-unsigned-ipa` (good for testing, URLs are not stable).
- Every tag matching `v*` attaches `Nimbus3D.ipa` as a GitHub Release asset. That asset is the stable `downloadURL` Bottle should use.

## The exact source.json entry

Add this object to the `apps[]` array in the file at `config.sideload.sourceJsonPath`
(see BOTTLE_INTEGRATION.md step 1; Bottle's own entry stays at whatever index it has,
and `resolveUnsignedIpa()` must select by `bundleIdentifier`, step 2):

```json
{
  "name": "Nimbus3D",
  "bundleIdentifier": "com.nimbus3d.app",
  "developerName": "Nimbus3D",
  "version": "0.1.0",
  "downloadURL": "https://github.com/OWNER/REPO/releases/latest/download/Nimbus3D.ipa"
}
```

### Placeholder rules

- `downloadURL`: `OWNER/REPO` is a PLACEHOLDER. The Nimbus3D app repo does not have a
  GitHub remote yet. Once it is pushed and the first `v*` tag builds, substitute the real
  owner and repo name. The `releases/latest/download/Nimbus3D.ipa` form is intentional:
  it always resolves to the newest release asset, so `source.json` never needs editing
  for version bumps. Pin a specific version with
  `releases/download/v0.1.0/Nimbus3D.ipa` instead if Bottle wants reproducible installs.
- `bundleIdentifier`: `com.nimbus3d.app` is the value in `app/project.yml`
  (`PRODUCT_BUNDLE_IDENTIFIER`). BOTTLE_INTEGRATION.md marks it "final value TBD";
  treat this file as the source of truth and re-check it against the first real
  CI-built IPA (`Payload/Nimbus3D.app/Info.plist`, key `CFBundleIdentifier`) before wiring.
- `version`: matches `MARKETING_VERSION` / `CFBundleShortVersionString` in
  `app/project.yml` (currently `0.1.0`). Bump together with tags.

## Signing notes for the Bottle side

- Sign Nimbus3D WITHOUT the Network Extension entitlement. The NE tunnel is
  Bottle-specific; Nimbus3D does not use it (BOTTLE_INTEGRATION.md step 3).
- Nimbus3D ships a large Metal/GPU payload plus an embedded Rust xcframework
  (`NimbusSplatCore`). Expect an IPA far larger than Bottle's. Check
  `config.sideload.maxIpaBytes` and the agent's `MAX_IPA_BYTES` (512 MB).
  The CI workflow emits a warning annotation if the IPA exceeds 512 MB.
- Free Apple ID limits (3 sideloaded apps, 7-day expiry) apply exactly as described
  in BOTTLE_INTEGRATION.md; Bottle's refresh loop must cover Nimbus3D too.

## Status

The workflow exists and is correct as written. The SplatEngine build script
`app/Sources/SplatEngine/build-xcframework.sh` (which produces
`Frameworks/NimbusSplatCore.xcframework`) now exists, so CI clears the workflow's
file-presence guard at the "Build Rust core" step. A green build now depends on the
Brush-based Rust crate actually compiling for the iOS targets (Brush is pulled from git
`branch = main` and is not yet pinned to a commit), which is SplatEngine's responsibility,
not CI's. Wire Bottle with the placeholder now; swap in the real `downloadURL` after the
first tagged green build.
