# Prompt for the Bottle Claude session

Paste everything below the line into a Claude Code session opened in the Bottle
project. It is written to stand alone: that session has no context from the
Nimbus3D work and should not be assumed to.

Every fact in the "verified" section was read out of the actual built artifact on
2026-09-05, not inferred from the build config.

---

## The ask

Add **Nimbus3D** as a second installable app in Bottle. Reuse the existing free
Apple ID re-sign and install pipeline unchanged, and add an "Install Nimbus3D"
button in the Tauri agent UI that installs it the same way Bottle installs
itself.

## Why this should be a small change

Bottle's sign-and-install pipeline is already app-agnostic: it is parameterised by
`bundle_id`, `udid` and `ipa_url`. `zsign`, the agent's `install_ipa` (usbmuxd ->
AFC -> installation_proxy Upgrade) and the relay host-muxer path need **zero**
changes. Only the app *selection* and some hardcoded UI copy are tied to Bottle.

(That reading of the pipeline came from `broker/routes/sideload.js` and
`agent/Windows/crates/bottle-agent/src/sideload.rs` as they stood on 2026-07-09.
Re-check both against the current source before trusting it: they may have moved.)

## The app, verified

These were read directly out of the built `.ipa`, so they are facts, not intentions:

| | |
|---|---|
| Bundle id | `com.tombline.nimbus` |
| Product / display name | `Nimbus3D` (a WORKING NAME, see the rename note below) |
| Version | `0.1.0` (`CFBundleShortVersionString`), build `1` |
| IPA size | **1,532,113 bytes (1.5 MB)** |
| Binary | `Payload/Nimbus3D.app/Nimbus3D`, Mach-O `MH_MAGIC_64`, cputype **arm64**, 3,287,720 bytes |
| Shaders | `Payload/Nimbus3D.app/default.metallib`, 299,112 bytes |
| Minimum iOS | 17.0 |
| Device family | `[1]` (iPhone only) |
| Required capabilities | `arkit`, `metal`, `arm64` |
| Bonjour | `NSBonjourServices: ["_nimbusboost._tcp"]` |
| Built with | GitHub Actions, macos-15, Xcode 16F6, iOS SDK 18.5 |

**Correction to any older note you may find:** earlier Nimbus3D docs warned that
this app would be "far larger than Bottle" and told you to check
`config.sideload.maxIpaBytes` and the agent's `MAX_IPA_BYTES` (512 MB). That was
written before the first build existed. The real IPA is **1.5 MB**. Size limits
are a non-issue; ignore that warning.

## THE ONE THING THAT WILL BLOCK YOU

`bottlemaster-22/Nimbus3D` is a **PRIVATE** repository. The release asset exists:

```
https://github.com/bottlemaster-22/Nimbus3D/releases/download/v0.1.0/Nimbus3D.ipa
```

but an unauthenticated GET of it returns **HTTP 404** (tested, not assumed).
Whatever downloads the IPA today almost certainly does an unauthenticated fetch,
so it will fail with a confusing 404 rather than a clear auth error.

Pick one, in the owner's preference order:

1. **Local file first (fastest path to a working install).** Let the Nimbus3D
   entry take a local path instead of a URL, and have the owner drop
   `Nimbus3D.ipa` somewhere Bottle can read. This proves the sign-and-install
   path end to end without touching auth at all. Do this first if the goal is
   simply to get the app onto the phone today.
2. **Authenticated download.** Send `Authorization: Bearer <token>` and
   `Accept: application/octet-stream` against the **API** asset URL
   (`https://api.github.com/repos/bottlemaster-22/Nimbus3D/releases/assets/<id>`),
   not the browser URL. A fine-grained PAT with read-only Contents scope on that
   one repo is enough. Store it the way Bottle already stores secrets; do not put
   it in `source.json`.
3. **Make the repo public.** The owner's call, not yours. Do not do this on your
   own initiative.

## The `source.json` entry

Add this to the `apps[]` array in the file at `config.sideload.sourceJsonPath`.
Bottle's own entry stays where it is.

```json
{
  "name": "Nimbus3D",
  "bundleIdentifier": "com.tombline.nimbus",
  "developerName": "TOMBLINE",
  "version": "0.1.0",
  "downloadURL": "https://github.com/bottlemaster-22/Nimbus3D/releases/latest/download/Nimbus3D.ipa"
}
```

`releases/latest/download/...` always resolves to the newest release, so this
never needs editing for version bumps. Swap it for a local path if you take
option 1 above, or pin `releases/download/v0.1.0/Nimbus3D.ipa` if you want
reproducible installs.

## Concrete changes on the Bottle side

1. **`source.json`**: add the entry above. Today `resolveUnsignedIpa()` only ever
   reads `source.apps[0]`.

2. **`resolveUnsignedIpa()`**: select by bundle id instead of hardcoding index 0.
   Thread a `bundleId` argument in from `runRefreshJob({ ... bundleId ... })`
   (it already has `bundleId` in scope) and match
   `source.apps.find(a => a.bundleIdentifier === bundleId)`.

3. **Entitlements, per app.** `runRefreshJob` injects the Network Extension
   entitlement when `config.sideload.injectNetworkExtensionEntitlement` is on.
   That NE tunnel is Bottle-specific and Nimbus3D does not use it. Gate the
   injection on the bundle id so Nimbus3D is signed **without** it.

4. **DB keying, the one real gotcha.** `sideload_identity` and `sideload_state`
   are unique on `(account_id, udid)`, i.e. Bottle assumes ONE app per device. To
   keep Bottle installed *and* add Nimbus3D on the same iPhone, add `bundle_id`
   to the uniqueness key on both tables (a migration). The Apple sign-in, cert
   and pairing record can stay shared across both apps for the same Apple ID;
   only the per-app state (which IPA, which expiry) needs its own row.

5. **UI button.** Add "Install Nimbus3D" beside the existing install/refresh
   control. It calls the same flow as the Bottle install:
   `POST /sideload/build` with `{ udid, bundle_id: "com.tombline.nimbus" }`.
   No new transport.

6. **Copy.** Hardcoded strings ("Refreshing Bottle", "Bottle is ready to
   install", the signed-artifact name `Bottle-signed-...`) should take the app
   name so a Nimbus3D install reads correctly.

## Signing notes

- Sign **without** the Network Extension entitlement (see change 3).
- Nimbus3D's `ios/project.yml` requests
  `com.apple.developer.kernel.increased-memory-limit` and
  `-extended-virtual-addressing`. **A free personal Apple ID team cannot grant
  either.** Expect your re-sign to strip them, or the install to fail
  specifically on entitlements. If it fails, strip them: the app is written to
  measure its real memory ceiling at runtime rather than assume the entitlement
  was granted, so it degrades instead of breaking. This is a Nimbus3D-side design
  decision, not something you need to work around.
- Free Apple ID limits apply exactly as they do for Bottle: **7-day expiry**, max
  3 sideloaded apps and 10 App IDs per 7 days. Bottle + Nimbus3D = 2 apps, fine.
  Bottle's existing refresh loop must now cover Nimbus3D too, or it will silently
  expire after a week.

## If the product gets renamed

"Nimbus3D" is a working name and may change. It comes from a single
`settingGroups.brand` block in `ios/project.yml`. If it changes, the `.ipa`
filename, `apps[].name` and the `downloadURL` all change with it. The bundle id
`com.tombline.nimbus` is settled and will not change.

## What this app is, briefly

A LiDAR 3D-scanning app: it records a room with the camera and the LiDAR sensor,
then trains a Gaussian splat model **on the phone** to produce a 3D scene. The
`_nimbusboost._tcp` Bonjour service is for an optional PC helper on the same
Wi-Fi that can take a large training job off the phone. None of that affects
sideloading; it is only context for why the app wants ARKit, Metal and local
network permissions.

## Honest status

This build **compiles and packages**. It has **never been run on a phone**. The
first install may well crash at ARKit session start or Metal pipeline setup. That
is expected and is the Nimbus3D side's problem to fix, not Bottle's. Your job is
done when the app installs and launches far enough to show a screen. If it
crashes on launch, capture the crash log and hand it back rather than trying to
fix the app.
