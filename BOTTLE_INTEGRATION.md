# Bottle integration: add an "Install Nimbus3D" button

_Hand this file to the Bottle Claude session. It is written to stand alone._
_Grounded in a read of `Bottle/broker/routes/sideload.js` and `Bottle/agent/Windows/crates/bottle-agent/src/sideload.rs` on 2026-07-09._

## The one-sentence ask
Add Nimbus3D as a **second installable app** in Bottle: reuse the existing free-Apple-ID re-sign + install pipeline unchanged, and add an "Install Nimbus3D" button in the Tauri agent UI that installs it the same way Bottle installs itself.

## Why this is a small change
The whole sign + install pipeline is already app-agnostic: it is parameterized by `bundle_id`, `udid`, and `ipa_url`. `zsign`, the agent's `install_ipa` (usbmuxd -> AFC -> installation_proxy Upgrade), and the relay host-muxer path need **zero** changes. Only the app *selection* and some UI copy are hardwired to Bottle.

## What Nimbus3D provides (the other side of the contract)
- An **unsigned** `Nimbus3D.ipa`, built by GitHub Actions (macOS runner, Xcode archive with `CODE_SIGNING_ALLOWED=NO`, then zipped as `Payload/Nimbus3D.app` into an `.ipa`). Bottle does the signing, so CI needs **no Apple certs**.
- A stable **downloadURL** for that IPA (a GitHub Release asset on the Nimbus3D repo).
- A **bundle id**: `com.tombline.nimbus`. This is now settled, not a placeholder: it is `NIMBUS_BUNDLE_ID` in the brand block of `ios/project.yml` and it is what the archive is built with. (An earlier draft of this file said `com.nimbus3d.app`; that value was never built and must not be used. `BOTTLE_HOOKS.md` already carries the correction.)
These three values are all Bottle needs. Until the first build exists, wire everything with a placeholder URL/bundle id.

## Concrete changes on the Bottle side

1. **`source.json`** (`config.sideload.sourceJsonPath`): add a second entry to `apps[]` for Nimbus3D with its `bundleIdentifier` + `downloadURL` (the GH release asset). Today `resolveUnsignedIpa()` only ever reads `apps[0]`.

2. **`broker/routes/sideload.js` -> `resolveUnsignedIpa()`**: make it select the app by bundle id instead of hardcoding `source.apps[0]`. Thread a `bundleId` argument in from `runRefreshJob({ ... bundleId ... })` (it already has `bundleId` in scope) down to `resolveUnsignedIpa(bundleId)`, and match `source.apps.find(a => a.bundleIdentifier === bundleId)`.

3. **Entitlements**: `runRefreshJob` injects the Network-Extension entitlement when `config.sideload.injectNetworkExtensionEntitlement` is on (line ~800). That NE tunnel is Bottle-specific. Make it **per-app** so Nimbus3D is signed WITHOUT the NE entitlement (Nimbus3D does not need Bottle's tunnel). Simplest: only inject NE when `bundleId === <Bottle's bundle id>`.

4. **DB keying (the one real gotcha)**: `sideload_identity` and `sideload_state` are unique on `(account_id, udid)`, i.e. Bottle assumes ONE app per device. To keep Bottle installed AND add Nimbus3D on the same iPhone, add `bundle_id` to the uniqueness key (migration on both tables) so the two apps get independent identity/state rows. The Apple sign-in / cert / pairing record can be shared across both apps for the same Apple ID; only the per-app state (which IPA, which expiry) needs to be separate.

5. **UI button** (Tauri agent frontend): add "Install Nimbus3D" next to the existing install/refresh control. It calls the same flow the Bottle install uses: `POST /sideload/build` with `{ udid, bundle_id: "<nimbus3d bundle id>" }` (agent-authed; signs the latest unsigned Nimbus3D IPA and installs over USB). No new transport.

6. **Copy**: the hardcoded strings ("Refreshing Bottle", "Bottle is ready to install", signed-artifact name `Bottle-signed-...`) should be app-named so Nimbus3D installs read correctly.

## Honest limits to know
- **7-day expiry**: installs use the free personal-team cert, so Nimbus3D (like Bottle) expires every ~7 days and Bottle's existing refresh loop must cover it too.
- **Free Apple ID caps**: max 3 sideloaded apps and 10 App IDs per 7 days per free Apple ID. Bottle + Nimbus3D = 2 apps, fine.
- **Nimbus3D IPA size**: this is a heavy Metal/GPU app, likely far larger than Bottle. Check `config.sideload.maxIpaBytes` and the agent's `MAX_IPA_BYTES` (512 MB) are big enough.

## Sequencing
This button depends on Nimbus3D existing as an IPA. Do the plumbing now with a placeholder; the real `downloadURL` + `bundle_id` land once Nimbus3D's first CI build publishes a release asset.
