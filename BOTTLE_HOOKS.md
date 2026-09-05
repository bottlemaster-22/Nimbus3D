# Bottle hooks: the exact values Bottle needs from Nimbus3D

_Companion to `BOTTLE_INTEGRATION.md` (same directory), which describes the Bottle-side
code changes. This file is the Nimbus3D-side contract: the concrete `source.json` entry
and where the IPA comes from. Hand both files to the Bottle session together._

_Note: BOTTLE_INTEGRATION.md was written against an earlier scaffold (Rust/Brush
core under `app/`, bundle id `com.nimbus3d.app`, working repo name TBD) and has not
been re-synced. That earlier scaffold was set aside to `_superseded/app-brush-wrapper/`
(see `SESSION_JOURNAL.md`); the app is now pure Swift + Metal under `ios/`, no Rust,
no Brush. This file (BOTTLE_HOOKS.md) reflects the CURRENT app and is the source of
truth for the values below — where BOTTLE_INTEGRATION.md's mechanics (steps 1-6 for
the Bottle-side code changes) still apply as written, its specific IPA/bundle-id/repo
details do not._

## Where the IPA comes from

`.github/workflows/ios.yml` in this repo builds an UNSIGNED `.ipa` on a macOS
runner: XcodeGen generates `ios/App.xcodeproj` from `ios/project.yml`, then
`xcodebuild archive` with `CODE_SIGNING_ALLOWED=NO` / `CODE_SIGNING_REQUIRED=NO`
and empty signing identities, then the `.app` is hand-zipped as
`Payload/<Product>.app` into `<Product>.ipa` (no Rust build step; `packages: {}`
in project.yml, nothing to cross-compile). There are no Apple certs in CI;
Bottle re-signs at install time with the user's free Apple ID.

The product name is not "App" (that is only the Xcode project/target/scheme
name, chosen to be brand-neutral). It is `NIMBUS_PRODUCT_NAME` from the
`settingGroups.brand` block in `ios/project.yml`, currently **`Nimbus3D`** — a
working name, not final (see `ios/Sources/Core/BrandConfig.swift`). The
workflow discovers it at build time by globbing
`Products/Applications/*.app` rather than hardcoding it, so a rename changes
only `ios/project.yml` and needs no CI edit. **If the product is renamed, the
`.ipa` filename and the `apps[].name` below change with it** — re-check this
file's JSON block against the actual artifact/release-asset name after a
rename, not just the text around it.

- Every push to `main` uploads the IPA as a workflow artifact named `<Product>-unsigned-ipa` (currently `Nimbus3D-unsigned-ipa`; good for testing, artifact URLs are not stable and expire in 30 days).
- Every tag matching `v*` attaches `<Product>.ipa` (currently `Nimbus3D.ipa`) as a GitHub Release asset. That asset is the stable `downloadURL` Bottle should use.
- A second, unrelated CI job (`booster-check`) lints the Python PC Booster under `booster/`. It does not touch the IPA and Bottle does not need to know about it.

## The exact source.json entry

Add this object to the `apps[]` array in the file at `config.sideload.sourceJsonPath`
(see BOTTLE_INTEGRATION.md step 1; Bottle's own entry stays at whatever index it has,
and `resolveUnsignedIpa()` must select by `bundleIdentifier`, step 2):

```json
{
  "name": "Nimbus3D",
  "bundleIdentifier": "com.tombline.nimbus",
  "developerName": "TOMBLINE",
  "version": "0.1.0",
  "downloadURL": "https://github.com/bottlemaster-22/Nimbus3D/releases/latest/download/Nimbus3D.ipa"
}
```

### Placeholder rules

- `downloadURL`: the owner/repo (`bottlemaster-22/Nimbus3D`) is REAL — the repo exists
  at `https://github.com/bottlemaster-22/Nimbus3D` — but the filename and the release
  itself are still a PLACEHOLDER until the first `v*` tag actually builds and publishes
  a Release asset. The `releases/latest/download/Nimbus3D.ipa` form is intentional: it
  always resolves to the newest release asset, so `source.json` never needs editing for
  version bumps. Pin a specific version with `releases/download/v0.1.0/Nimbus3D.ipa`
  instead if Bottle wants reproducible installs. If the product is later renamed away
  from "Nimbus3D", this filename (and the `apps[].name` above) must be updated to match
  — see "Where the IPA comes from" above.
- `bundleIdentifier`: `com.tombline.nimbus` is `NIMBUS_BUNDLE_ID` in
  `ios/project.yml` (`settingGroups.brand`, feeds `PRODUCT_BUNDLE_IDENTIFIER`) and
  the single source of truth read at runtime by `ios/Sources/Core/BrandConfig.swift`.
  This supersedes `com.nimbus3d.app`, the placeholder BOTTLE_INTEGRATION.md's step 1-2
  text still shows — that value was never real; TOMBLINE is the reverse-DNS owner, not
  "nimbus3d". Re-check this against the first real CI-built IPA
  (`Payload/Nimbus3D.app/Info.plist`, key `CFBundleIdentifier`) before wiring.
- `version`: matches `NIMBUS_MARKETING_VERSION` / `CFBundleShortVersionString` in
  `ios/project.yml` (currently `0.1.0`). Bump together with tags.

## Signing notes for the Bottle side

- Sign Nimbus3D WITHOUT the Network Extension entitlement. The NE tunnel is
  Bottle-specific; Nimbus3D does not use it (BOTTLE_INTEGRATION.md step 3).
- Nimbus3D ships a large Metal/GPU payload (on-device 3DGS training data,
  submaps, LiDAR sidecars). There is no embedded Rust xcframework — the earlier
  Brush/Rust core was set aside; the app is pure Swift + Metal. Still expect an
  IPA larger than Bottle's own. Check `config.sideload.maxIpaBytes` and the
  agent's `MAX_IPA_BYTES` (512 MB); the CI workflow emits a `::warning::`
  annotation if the packaged IPA exceeds 512 MB.
- `ios/project.yml`'s entitlements request
  `com.apple.developer.kernel.increased-memory-limit` and
  `-extended-virtual-addressing` for large-scan training headroom. A free
  personal Apple ID team cannot grant these; Bottle's re-sign will either strip
  them or the install will fail on signing. The app must never assume they were
  granted (see `TrainingBudget`, which measures the real ceiling at runtime) —
  this is a Nimbus3D-side concern, not something Bottle needs to work around,
  but worth knowing if a sideload install fails specifically on entitlements.
- Free Apple ID limits (3 sideloaded apps, 7-day expiry) apply exactly as described
  in BOTTLE_INTEGRATION.md; Bottle's refresh loop must cover Nimbus3D too.

## Status

**SUPERSEDED IN PART BY `BOTTLE_PROMPT.md`** (same directory), which is the
self-contained prompt to hand to the Bottle session and carries the verified
post-build facts. Read that first; this file remains the Nimbus3D-side reference.

**CI HAS NOW RUN AND SUCCEEDED.** Run 33973412814 on 2026-09-05 produced a real
unsigned IPA on macos-15 (Xcode 16F6, iOS SDK 18.5), after four rounds fixing 27
compile errors. Verified by reading the artifact itself:

- `Payload/Nimbus3D.app/Nimbus3D` - Mach-O `MH_MAGIC_64`, cputype arm64, 3,287,720 bytes
- `Payload/Nimbus3D.app/default.metallib` - 299,112 bytes, so all three .metal files compiled AND linked
- `CFBundleIdentifier` = `com.tombline.nimbus`, confirming the value this file predicted
- `MinimumOSVersion` 17.0, `UIDeviceFamily` [1], `UIRequiredDeviceCapabilities` [arkit, metal, arm64]

**The release asset now EXISTS** and is no longer a placeholder:
`https://github.com/bottlemaster-22/Nimbus3D/releases/download/v0.1.0/Nimbus3D.ipa`

**CORRECTION to the signing notes above:** this file warned that Nimbus3D "ships a
large Metal/GPU payload" and to check `maxIpaBytes` / `MAX_IPA_BYTES` (512 MB).
The real IPA is **1.5 MB**. Size limits are a non-issue. The warning was written
before a build existed and was wrong.

**NEW BLOCKER this file could not have known:** `bottlemaster-22/Nimbus3D` is a
PRIVATE repo, so an unauthenticated GET of the release asset returns HTTP 404
(tested). `BOTTLE_PROMPT.md` lists the three ways to handle that.

**Still true:** the app compiles but has NEVER RUN on hardware.
