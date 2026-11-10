# Nimbus3D

An iPhone app that walks around a room with you, measures it with the laser
scanner built into the phone, and turns what it saw into a 3D model you can
look at, spin around, and export.

"Nimbus3D" is a working name. It appears in exactly one place in the code (the
brand block at the top of `ios/project.yml`) and everything else reads it from
there, so renaming the whole product is a five line edit.

---

## What it actually does

The phone has two things pointed at the room: a camera and a small laser
scanner (Apple calls it LiDAR). The camera sees colour. The laser measures
distance, honestly, in metres, about fifty thousand times a second. Most apps
of this kind throw the laser away and try to guess distance from the photos.
This one does the opposite: the laser is the evidence, and the photos are used
to make it look right.

That choice runs through the whole app:

* **It knows the difference between "empty" and "unknown."** If a laser beam
  passed through a bit of air and came back off the wall behind it, that air is
  known to be empty and anything the model invented there gets deleted. If the
  beam came back with nothing (a window, a mirror, something too far away),
  that space is unknown, and unknown is left alone. Guessing there is how you
  get a scan with the garden erased.
* **It shows you what it made up.** Parts of the model you never actually
  looked at from any angle are drawn with diagonal hatching. A tool tells you
  where it is weak. A toy hides it.
* **It scores itself against photos it was not allowed to see.** Roughly one
  photo in twenty is held back from training, and the comparison slider uses
  those. Comparing against a photo the model was trained on proves nothing.
* **Nothing leaves your phone.** No account, no cloud, no analytics, no
  outbound connection anywhere. The one network feature (the PC Booster, below)
  talks only to a computer on your own Wi-Fi, and it is entirely optional.

**It needs an iPhone with a laser scanner.** Apple has only fitted one to the
Pro and Pro Max models, starting with iPhone 12 Pro. The app does not go by a
list of model names: on first launch it asks the phone itself whether it can
do laser scanning, and if the answer is no it says so in plain words rather
than letting you waste a walk around your living room on a scan that cannot
work. There is no way to fake that measurement from the camera alone, and
pretending otherwise would be the dishonest option.

Requires iOS 17 or later. iPhone only, no iPad.

---

## How it is put together

Everything under `ios/Sources/` is Swift and Metal. No Rust, no third party
libraries, no package manager. That is deliberate: it means there is nothing
for a build machine to download, resolve, or fail on.

```
ios/Sources/
  Core/         the shared vocabulary every other folder speaks
  App/          the app shell: what launches, what is on which tab
  Onboarding/   the first-launch check: can this iPhone do this at all
  Capture/      the scanning screen and everything it writes to disk
  PrePass/      the fast pass that checks a scan over before any training
  Smart/        the pieces that make it smarter than a generic splat trainer
  Trainer/      the 3D model builder, written from scratch in Metal
  Viewer/       looking at a finished scan, and exporting it
  Export/       reading and writing .ply, .spz and .glb files
  Booster/      finding and talking to a PC on your Wi-Fi
booster/        the optional PC helper (Python), not needed to use the app
blender_addon/  opens a finished scan in Blender
docs/           the file format and the PC protocol, written down exactly
```

Three reference documents are treated as law rather than as notes:
`ios/Sources/Core/Contracts.swift` (every shared type), `docs/DATA_FORMAT.md`
(every filename and byte layout on disk) and `docs/BOOSTER_PROTOCOL.md` (every
message between phone and PC). If code and one of those disagree, the document
is right and the code is a bug.

---

## Building the app

There is no Xcode project file checked in. It is generated, so that adding a
file is never a merge conflict and renaming the product never means editing
twenty places.

### The normal way: let GitHub build it

Push to `main` and the workflow in `.github/workflows/ios.yml` does all of it
on a Mac that GitHub rents you:

1. installs XcodeGen and generates `App.xcodeproj` from `ios/project.yml`
2. archives the app with signing switched off (there are no Apple
   certificates anywhere in this repository, and there should not be)
3. packages the result by hand into an **unsigned** `Nimbus3D.ipa`
4. uploads that file as a build artifact on every run

Push a tag starting with `v` (for example `v0.1.0`) and it also attaches the
`.ipa` to a GitHub Release. That Release link is what the Bottle installer
points at; see `BOTTLE_INTEGRATION.md`.

**Unsigned means it cannot be installed by double clicking it.** Something has
to sign it with your Apple ID first. That is what Bottle does. A free Apple ID
signature expires after seven days and has to be refreshed, which is Apple's
rule, not this project's.

### If you have a Mac in front of you

```
brew install xcodegen
cd ios
xcodegen generate
open App.xcodeproj
```

Then press Run with your iPhone plugged in. Xcode will ask for a signing team
the first time; a free personal Apple ID is enough.

Two settings in `ios/project.yml` are worth knowing about before you change
anything:

* `SWIFT_VERSION` is `"5.0"`. This is the language mode, not the compiler
  version. `"5.10"` is not a legal value there and makes the build fail
  outright, which is a confusing hour to lose.
* The entitlements ask for a raised memory limit, which large scans need. A
  free Apple ID cannot grant it. The trainer measures the real limit at
  runtime and works within whatever it actually got, so if signing complains,
  deleting those two lines costs you nothing else.

### Renaming the product

Edit the five values in the `brand` block at the top of `ios/project.yml` and
nothing else, anywhere. One rule: the short slug has to be nine characters or
fewer, because it becomes part of a network name that Apple caps at fifteen
characters including punctuation. Break that and the phone silently stops
finding your PC, with no error message at all.

---

## The PC Booster (optional)

A whole house is a lot to ask of a phone. The Booster lets a computer on the
same Wi-Fi do the heavy part instead. You never need it, and the app never
requires it.

```
cd booster
pip install -e .[gui]
nimbus-booster-gui
```

The window shows a six digit code. Open the Booster tab on the phone, tap your
PC, type the code once, and they are paired for good. After that the phone can
hand a scan over, watch it work, and pull the result back. Scans land in
`Documents/Nimbus3D/Scans/` on the PC, and you can point that somewhere else in
the window.

For the PC to actually build a model, rather than just receive and keep a scan,
it needs an NVIDIA graphics card:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install gsplat
```

If it does not have one, the Booster still runs, still receives your scan, and
tells you in plain words that this PC cannot build it. It does not fail
silently and it does not pretend.

Full detail is in `booster/README.md`.

---

## What is not finished

Written down plainly, because a list of what works is worthless without it.

**1. Nothing on the phone starts the model builder yet.** This is the big one.
The pre-pass and the trainer are both written, both real, and both connected to
the app. What does not exist is the screen with the button: something that
takes a scan you just recorded, runs the check-over, then runs the trainer, and
shows you how it is going. The scan list even says "Ready to build the 3D
model" and there is nothing to tap.

Until that screen exists, the way to get a finished model out of a scan is to
send it to the PC Booster.

**2. The PC Booster does not sharpen the model yet.** It receives your scan and
turns it into real geometry from the laser measurements, and gives you back
files you can open. What it does not do yet is the optimisation pass that makes
a splat model look photographic rather than like a very good point cloud. That
code has not been written. What comes back is honest, it is just not as sharp
as it will be.

**3. One thing on the phone is a labelled placeholder.** For surfaces in the
awkward middle distance, past the laser's confident range but not far enough
away to be treated as background, the design calls for a depth estimate from a
small machine learning model. No such model is bundled, because no suitable one
has been published in a form iOS can load. The placeholder returns nothing and
says so, rather than inventing a number. Everything else on the phone is real
code doing real work.

**4. None of the Swift has ever been compiled.** It was written on a Windows
machine, so there was no Xcode to check it. Every cross-module reference,
shader name and type name was swept mechanically against the actual
declarations, which catches a great deal, but it is not the same as a compiler
agreeing. The first GitHub Actions run on a Mac is the first real test, and it
should be expected to find things.

**5. Reading `.spz` version 4 files is refused, not guessed.** Versions 1 to 3
are read and written for real. Nobody has published a version 4 sample to test
against, so the reader says it cannot open one instead of producing something
subtly wrong.

Per module detail, with the same honesty, is in `MODULE_STATUS.md`.
