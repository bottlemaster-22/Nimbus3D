# App Store readiness

The owner's stated end goal (2026-09-06): **"I would like it to be App-Store ready by
the end of making the app."** Not a priority now. This file exists so the decisions
that are expensive to reverse get made correctly while the app is still being built,
rather than discovered during a submission.

Nothing in here should slow down fixing the app. It is a checklist to keep honest,
not a workstream.

---

## Hard blockers (submission is refused without these)

### 1. There is no app icon. At all.
`ios/project.yml` declares no icon asset and there is no `Assets.xcassets` anywhere in
the repo, so the app currently installs with the grey placeholder. This is the thing
the owner actually noticed.

What is needed: a single **1024x1024 PNG, no alpha channel, no transparency, no
rounded corners** (Apple applies the mask). Modern Xcode generates every other size
from it. It goes in an asset catalogue with `ASSETCATALOG_COMPILER_APPICON_NAME` set
in the build settings.

**Needs a design decision from the owner, not a default.** Do not invent branding.

### 2. `PrivacyInfo.xcprivacy` is missing and is MANDATORY.
Apple has required a privacy manifest for App Store submission since May 2024. It
declares data collection and, critically, the **required-reason APIs** the binary
calls. This app uses several:

| API category | Where | Reason code |
|---|---|---|
| File timestamps | scan folders, census files, `ScanLibraryStore` | `C617.1` (app-container files) |
| Disk space | capture storage guard, `TrainingBudget` | `E174.1` (write-failure avoidance) |
| `UserDefaults` | `DeviceReportStore`, onboarding completion, Booster pairing | `CA92.1` (app-container only) |

Cheap to write, and cheap to keep correct while the code is fresh. Retrofitting it
means auditing 125 files for API calls nobody remembers making.

### 3. NAME DECIDED 2026-09-06: **LiKOVA**

The owner chose it after four candidates died on checks. His reasoning, and it
answers the objection raised against it:

- Pronunciation ambiguity (LIE-kova vs LEE-kova) is ACCEPTABLE to him. Intended
  is "Lie-KOH-vah", and he considers "Lee-KOH-vah" fine too. Precedent: Lichess,
  which has the same ambiguity and does not suffer for it.
- He judges LiKOVA genuinely distinguishable from the bare "Kova" that the UK
  register killed.

Root: Finnish `kova`, "hard, solid, firm", from Proto-Finnic `*kova`, with
cognates in Finnish, Karelian, Ingrian, Estonian (`kõva`) and Livonian. A
scanner's whole job is separating solid from empty, which is also exactly what
the free-space carving feature does, so the etymology is defensible in one
sentence rather than invented.

STILL TO CHECK BEFORE ANY ART IS COMMISSIONED: run the UKIPO and USPTO searches
on "LiKOVA" filtered to classes 9 and 42, the same search that killed Kova. Bare
KOVA has at least three LIVE UK class 9 registrations (UK00003341089,
UK00003377484, and UK00004205547 filed May 2025 for class 9 alone), so the
question is whether the Li prefix is enough separation. That is a real question
and it is unanswered.

BRAND BLOCK VALUES when the rename is made (ios/project.yml, settingGroups.brand):
  NIMBUS_PRODUCT_NAME: LiKOVA
  NIMBUS_DISPLAY_NAME: LiKOVA
  NIMBUS_BUNDLE_ID:    com.tombline.likova
  NIMBUS_BRAND_SLUG:   likova          (6 chars, inside the 9-char DNS-SD limit)
  NIMBUS_DOCS_FOLDER:  LiKOVA
Note the slug must stay lowercase a-z0-9: the Bonjour type becomes
`_likovaboost._tcp`, which is 14 characters including the underscore, inside the
15-character cap with one to spare.

### WHY THE PREVIOUS NAMES DIED (kept as the record)

Original verdict on Nimbus3D:

Not merely crowded. Taken by directly adjacent products:

- **pieye sells a product called "Nimbus 3D"** and it is a TIME-OF-FLIGHT DEPTH
  CAMERA with its own software (shop.pieye.org). That is the same sensing technology
  this app is built on, which is the worst possible kind of collision: same words,
  same field.
- **nimbus-3d.com** is a 3D rendering and visualisation studio.
- **NIMBUS is a registered software trademark several times over**: TIBCO Software,
  Timehop (SaaS), Athletai, Globecomm, LAN Control Systems, plus Nimbus Inc and
  NuVasive in other classes.
- On the App Store, "Nimbus Note" already occupies the name in a scanning-adjacent
  category (document scanning).

The original rationale is also half dead. The journal's first entry says it was
"named Nimbus3D (splat = cloud of blobs; nimbus also nods to sky/HDRI environment)".
The cloud-of-blobs half still holds. The HDRI half was cut at the reset, when research
showed a usable HDRI cannot be recovered from LDR splat colour. So the name is
carrying meaning the product no longer has.

**A rename is cheap right now and gets expensive fast.** The brand block in
`ios/project.yml` makes it one line, which was a deliberate decision made for exactly
this moment. It stops being cheap the second an icon and a wordmark are designed
around it.

CONSTRAINT ON ANY NEW NAME, from this project's own wire protocol: the brand slug
becomes the Bonjour service type `_<slug>boost._tcp`, and DNS-SD caps a service type
label at 15 characters INCLUDING the leading underscore. So the slug must be **9
characters or fewer**.

---

## Needs a paid developer account (99 USD/year)

- **`com.apple.developer.kernel.increased-memory-limit`** and
  `extended-virtual-addressing`, both requested in `project.yml`. A free Apple ID
  team **cannot grant either**, which is why Bottle strips them on re-sign, and is a
  live hypothesis for on-device memory kills today. On a paid account they are
  available and this app genuinely wants them.
- Real code signing, so the 7-day sideload expiry disappears.
- TestFlight, which is the sane way to test on hardware without the Bottle loop.

---

## Review-risk items specific to THIS app

- **ARKit and camera usage strings** are already written and are plain and specific,
  which is what review actually looks for. Currently correct, do not let them rot.
- **Local network permission** (the PC Booster). Review asks what it is for. The
  usage string already explains it. The Booster must remain genuinely optional: an
  app that appears broken without a companion PC is a rejection risk.
- **No account, no login, no analytics, no third-party SDKs.** This is a strong
  position for review and for the privacy manifest. Keep it.
- **Age rating**: 4+ is achievable. Nothing user-generated is shared.
- **Export compliance**: the app does no encryption of its own. It will still need
  the standard declaration.
- **Performance**: review runs on real hardware. A crash during review is a
  rejection, which is one more reason the crash and census work matters.

## Accessibility

The app is already unusually well placed here, by accident of design: audio and
haptics are the PRIMARY guidance channel during capture, because the user is looking
at the room rather than the screen. That is close to what VoiceOver users need. What
is missing is explicit accessibility labels on the HUD controls and the review
screens, and a check that the coverage colours are distinguishable without colour
(the three channels are already separated by their own fix hints, which helps).

---

## What is already right, and worth not breaking

- Rename-in-one-place brand block, so the name is not baked into 125 files.
- No third-party dependencies at all, so no licence audit and no supply chain.
- iPhone-only (`TARGETED_DEVICE_FAMILY: "1"`), which is allowed and avoids having to
  make an iPad layout work.
- Honest usage strings and an onboarding screen that tells an incompatible device
  exactly why, which is the kind of thing review reacts well to.
- CI already produces a signed-nothing archive, so switching to a real signing
  identity is a settings change rather than a rebuild.

---

## Order to do this in, when the time comes

1. Decide the **name**. Everything else depends on it and it is one line to change.
2. **Icon**, once the name is settled.
3. **Privacy manifest**, which can be written today and kept current.
4. Paid account, real signing, TestFlight.
5. Accessibility labels.
6. Screenshots, description, keywords, age rating, export compliance.

Steps 3 and 5 are the ones that get cheaper the earlier they happen. The rest can
wait until the app actually works.
