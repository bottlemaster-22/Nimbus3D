# Booster protocol

The wire contract between the phone (Swift client, `ios/Sources/Booster/`) and
the optional PC Booster (Python server, `booster/`).

This document is **normative**. Where it and any implementation disagree, the
implementation is wrong. Its purpose is that the Swift client and the Python
server can be written by different people, at different times, without ever
talking, and still interoperate on the first try.

## Status of this document

The Swift client was written **before** this document existed and had to invent
a protocol to be implementable at all. Rather than force a rewrite of ~2900
lines of working, reviewed code, this document was written to match that
client, byte for byte, with the reasoning behind each choice recorded so the
Python side can be judged against something rather than guessed at.

Everything below has been read off the actual client source:
`BoosterProtocol.swift`, `BoosterHTTP.swift`, `BoosterDiscovery.swift`,
`BoosterPairingManager.swift`, `BoosterUploadManager.swift`,
`BoosterDownloadManager.swift`, `BoosterJobMonitor.swift`. Nothing here is
aspirational. Where the client's behaviour is surprising, the surprise is
called out rather than smoothed over, because the surprise is what breaks a
server written from a tidier imaginary spec.

Nothing in this protocol is required for the app to work. The Booster is
optional, LAN-only, and there is no cloud anywhere in it.

---

## 1. Naming, and why nothing here is hardcoded

The product name is a working name. Every identifier below that contains it is
derived from the single brand block in `ios/project.yml`:

| Thing | Value today | Where it comes from |
|---|---|---|
| Bonjour service type | `_nimbusboost._tcp` | `NIMBUS_BRAND_SLUG` -> `_<slug>boost._tcp` |
| Bonjour domain | `local.` | `BrandConfig.boosterServiceDomain` |
| Default TCP port | `8760` | `BrandConfig.boosterDefaultPort` |
| PC save root | `Documents/Nimbus3D` | `NIMBUS_DOCS_FOLDER` |

The Python server must read these from its own single-source-of-truth constants
file and never from a string literal scattered through the code. A rename is
one line on each side.

DNS-SD caps a service-type label at 15 characters including the leading
underscore. `_nimbusboost` is 12. A longer brand slug breaks discovery
silently - `BrandConfig.assertConsistent()` asserts on this in debug builds.

---

## 2. Discovery

The Booster **advertises** a DNS-SD service of type `_nimbusboost._tcp` in
domain `local.` on the port it is listening on.

The phone **browses** with `NWBrowser` and shows every instance it finds. The
Bonjour *instance name* is what the user sees in the device list before
pairing ("Ollie's PC"), so name the service after the computer, not after the
product.

Resolution: the client resolves the service to a literal `host:port` only when
the user acts on it, by opening an `NWConnection` to the service endpoint and
reading `currentPath.remoteEndpoint`. Both IPv4 and IPv6 literals are handled.

### TXT record

The client **does not read the TXT record today**. Publish these keys anyway -
they cost nothing and a later client version will use them to show a paired
device's real name and to reject an incompatible Booster before the user taps
anything:

| Key | Value |
|---|---|
| `api` | API version string, `1` |
| `bid` | The Booster's stable `boosterID` (a UUID, generated once, persisted) |
| `name` | Human-readable Booster name |

Because the client ignores TXT, a device's identity before pairing **is** its
Bonjour instance name. After pairing, identity is the `boosterID` returned by
the pairing handshake, and the phone remembers the mapping from instance name
to `boosterID` so a paired PC is recognised the moment it reappears.

Consequence for the server: **`boosterID` must be stable across restarts.**
Generate it once, write it next to the config, never regenerate it. A Booster
that changes its id looks like a brand-new unpaired device every time it boots.

---

## 3. Transport

* Plain **HTTP/1.1** over TCP. Not HTTPS.
* One **WebSocket** (`ws://`, same host and port) for live job progress.
* All paths are under `/v1`.
* Request and response bodies are **UTF-8 JSON** unless stated otherwise.
* Binary chunks use `Content-Type: application/octet-stream`.

### Why no TLS

This is a link between two devices on the same home Wi-Fi, with no CA that
could vouch for either of them. A self-signed certificate would produce a green
padlock that means nothing, teach the user to click through certificate
warnings, and cost real complexity. The actual protection here is that the
server binds to the LAN interface only, that no request is honoured without a
bearer token, and that the token is issued only after a human read a code off
the PC screen and typed it into the phone.

The phone's `Info.plist` therefore does **not** need an ATS exception for this:
`NSLocalNetworkUsageDescription` plus the Bonjour service declaration is what
iOS requires, and `NWBrowser`/`URLSession` to a `.local` or private-range host
is not blocked by ATS. If a future iOS build does start blocking it, the fix is
an `NSAllowsLocalNetworking` exception, not a fake certificate.

### Headers

Every request from the phone carries:

```
X-Nimbus-Api-Version: 1
```

and, once paired, also:

```
Authorization: Bearer <token>
```

The server must echo `X-Nimbus-Api-Version: <its own version>` on **every**
response, and especially on a `426` - the client reads it to name the mismatch
in a plain-language message instead of saying "unknown".

### Timeouts (client side, for reference)

| Operation | Timeout |
|---|---|
| Control requests (info, pair, create, finalize, status) | 20 s |
| One upload chunk | 60 s |
| One download range | 60 s |

The client retries a failed chunk 3 times with 0.5 s / 1 s backoff before
giving up on the transfer.

---

## 4. JSON encoding rules

These are the details that actually break interoperability. Get them wrong and
everything parses fine right up until it does not.

1. **Field names are exactly the Swift property names**, camelCase, with no
   custom `CodingKeys` anywhere. `scanID`, `jobID`, `boosterID`,
   `relativePath`, `byteCount`, `sha256`, `receivedByteOffset`,
   `fractionComplete`. Note the capitalisation of `ID`: it is `scanID`, never
   `scanId`.

2. **Dates are ISO-8601 without fractional seconds.** The client uses
   Foundation's `JSONDecoder.DateDecodingStrategy.iso8601`, which is
   RFC 3339 **to whole seconds only**. `2026-09-03T14:12:05Z` decodes.
   `2026-09-03T14:12:05.512Z` **fails to decode** and takes the whole message
   with it. In Python: `datetime.now(timezone.utc).replace(microsecond=0)
   .isoformat().replace("+00:00", "Z")`.

3. **A non-optional field must be present.** Swift's synthesised decoding
   throws on a missing key. Fields typed `T?` below may be omitted or `null`;
   everything else is mandatory. Sending `null` for a non-optional field is an
   error, not a shortcut.

4. **Unknown extra fields are ignored** by the client, so the server may add
   fields additively without a version bump. Removing or renaming one is a
   breaking change.

5. **Enums are lowercase strings**, exactly the spellings listed in section 8.
   An unrecognised value fails the decode of the whole message; if the server
   ever needs a new stage, that is an API version bump.

6. **Integers are JSON numbers, not strings.** `byteCount` and
   `receivedByteOffset` are 64-bit signed. `sha256` is a lowercase hex string,
   64 characters, no `0x`.

---

## 5. Endpoints

| Method | Path | Body in | Body out |
|---|---|---|---|
| `GET` | `/v1/info` | - | `BoosterInfo` |
| `POST` | `/v1/pair/requests` | `PairRequest` | `PairRequestResponse` |
| `POST` | `/v1/pair/requests/{requestID}/confirm` | `PairConfirm` | `PairStatus` |
| `GET` | `/v1/pair/requests/{requestID}` | - | `PairStatus` |
| `POST` | `/v1/jobs` | `CreateJob` | `CreateJobResponse` |
| `PUT` | `/v1/jobs/{jobID}/files/{relativePath}` | raw bytes | `ChunkUploadResponse` |
| `POST` | `/v1/jobs/{jobID}/finalize` | - | `FinalizeResponse` |
| `GET` | `/v1/jobs/{jobID}` | - | `JobStatus` |
| `DELETE` | `/v1/jobs/{jobID}` | - | any 2xx, body ignored |
| `WS` | `/v1/jobs/{jobID}/stream` | - | stream of `ProgressEvent` |
| `GET` | `/v1/jobs/{jobID}/result/manifest` | - | `Manifest` |
| `GET` | `/v1/jobs/{jobID}/result/files/{relativePath}` | - | raw bytes (Range) |

`/v1/info` is the only endpoint that may be served without a token.

### `{relativePath}` routing

The client percent-encodes the relative path with iOS's `.urlPathAllowed`
character set, **which does not escape `/`**. A file at
`sensor_data/depth/frame_20260903_141205_512.depth16` therefore arrives as
real path segments:

```
PUT /v1/jobs/7f3a.../files/sensor_data/depth/frame_20260903_141205_512.depth16
```

The server route must be a catch-all wildcard (`/v1/jobs/<job_id>/files/<path:rel>`
in Flask terms), not a single segment.

**Path traversal is the obvious attack here and the server must refuse it.**
Reject any relative path that, after decoding, is absolute, contains a `..`
segment, contains a backslash, contains a NUL, or resolves outside the job
directory. Reject rather than sanitise: a caller sending `../` is not making a
typo.

---

## 6. Pairing

A deliberately human-witnessed handshake. No accounts, no cloud, no QR code,
no silent trust of whatever answers on port 8760.

```
phone                                   PC Booster
  |  POST /v1/pair/requests                  |
  |  {deviceID, deviceName, appVersion}      |
  |----------------------------------------->|
  |                                          |  window shows a 6-digit code
  |  {requestID, awaitingCode: true}         |
  |<-----------------------------------------|
  |                                          |
  |  (user reads the code off the PC and     |
  |   types it into the phone)               |
  |                                          |
  |  POST /v1/pair/requests/{id}/confirm     |
  |  {code: "418203"}                        |
  |----------------------------------------->|
  |  {status: "approved", token, boosterID,  |
  |   boosterName}                           |
  |<-----------------------------------------|
```

**`PairRequest`**

```json
{ "deviceID": "5C3A...-UUID", "deviceName": "Ollie's iPhone", "appVersion": "0.1.0 (1)" }
```

`deviceID` is a UUID generated once per app install and persisted; it is not
the device's real identifier and carries nothing about the user.

**`PairRequestResponse`**

```json
{ "requestID": "b2e9...", "awaitingCode": true }
```

**`PairConfirm`**

```json
{ "code": "418203" }
```

**`PairStatus`**

```json
{ "status": "approved", "token": "opaque-...", "boosterID": "…", "boosterName": "Studio PC" }
```

`status` is one of `pending`, `approved`, `denied`, `expired`.
`token`, `boosterID` and `boosterName` are optional and are present only when
`status` is `approved`. If `approved` arrives without a token the client shows
"The Booster approved pairing but did not send a token" and stops - so do not
do that.

Server requirements:

* The code is at least 6 digits, generated with a CSPRNG, shown only in the
  Booster's own window, and **expires after 2 minutes**.
* At most 5 wrong code attempts per request, then the request becomes
  `expired`.
* The token is at least 256 bits of CSPRNG output, hex or base64url, opaque to
  the client. One token per paired phone. Store hashed if you like; the client
  never sends it anywhere except this Booster.
* `GET /v1/pair/requests/{requestID}` polls the same state, for the case where
  the server models approval asynchronously (the user clicks Allow on the PC).
  The client polls it while the code-entry sheet is open and ignores errors.
* Pairing must be revocable from the Booster's GUI. A revoked token starts
  returning `401`, and the phone reports "not paired with that Booster yet".

The phone stores the token in the iOS Keychain under
`<bundle-id>.booster`, keyed by `boosterID`.

---

## 7. Sending a scan

### 7.1 The manifest

The client walks the scan folder, hashes every file, and describes the whole
bundle up front.

```json
{
  "scanID": "scan_20260903_141150",
  "files": [
    { "relativePath": "capture_bundle.json", "byteCount": 184320,
      "sha256": "9f2c…" },
    { "relativePath": "images/frame_20260903_141205_512.jpg",
      "byteCount": 421887, "sha256": "0ab3…" }
  ]
}
```

Files are sorted by `relativePath`, ascending, byte-wise. The order is stable
so a rebuilt manifest lines up with a partially-uploaded job.

`totalByteCount` is a **computed** property on the client and is therefore not
serialised. Do not expect it; do not require it. Sending it is harmless.

### 7.2 Create or resume a job

```
POST /v1/jobs
{ "manifest": { …as above… }, "appVersion": "0.1.0 (1)" }
```

```json
{
  "jobID": "7f3a…",
  "chunkSize": 1048576,
  "receivedByteOffsets": { "images/frame_20260903_141205_512.jpg": 1048576 }
}
```

**Jobs are keyed by `scanID`.** Posting the same `scanID` again must return the
**same `jobID`** and the bytes already on disk, per file. This is the entire
resume mechanism: there is no local resume-state file on the phone, so resume
survives an app kill, a crash, and a full reinstall, as long as the Booster
still holds the job.

`receivedByteOffsets` is mandatory. Send `{}` for a brand-new job, not `null`.
It may omit files with zero bytes received; a missing key is read as 0.

`chunkSize` is advisory and **the client ignores it**: it always sends 1 MiB
(1048576 bytes) windows, and the last chunk of a file is short. The server must
accept whatever size arrives. The field exists so a future client can honour it.

If the manifest for an existing `scanID` differs from the stored one (a file
changed size or hash), the server must treat that file's progress as void and
report offset 0 for it.

### 7.3 Upload chunks

```
PUT /v1/jobs/{jobID}/files/{relativePath}
X-Nimbus-Chunk-Offset: 1048576
X-Nimbus-Chunk-Sha256: <sha256 of THIS CHUNK, lowercase hex>
Content-Type: application/octet-stream

<raw bytes>
```

```json
{ "receivedByteOffset": 2097152 }
```

Rules:

* `X-Nimbus-Chunk-Sha256` is the hash of **the chunk**, not the file. Verify it
  and reject a mismatch with `409`.
* Chunks for one file arrive strictly in ascending offset order, one at a time.
* `receivedByteOffset` is the **total bytes now held for this file** and is
  **authoritative**. The client trusts it over its own bookkeeping: if it does
  not equal `offset + len(chunk)`, the client seeks its local file handle to
  the value you returned and continues from there. That makes a partial write
  on the server side recoverable instead of silently corrupting.
* A chunk whose offset is behind what you already hold should be discarded and
  answered with your current total, not appended.
* A chunk whose offset is *ahead* of what you hold is a bug on one side or the
  other: answer with your current total so the client rewinds.

### 7.4 Finalize

```
POST /v1/jobs/{jobID}/finalize
```

```json
{ "accepted": true }
```

or

```json
{ "accepted": false,
  "reason": "One of the picture files did not arrive correctly. Send it again." }
```

The server verifies every file's whole-file SHA-256 against the manifest and
that every file is complete. `reason` is shown to the user **verbatim**, so
write it as a plain sentence, not as a stack trace or an error code.

A rejected finalize leaves the job resumable: the client can re-`POST /v1/jobs`
and the offsets tell it what to re-send.

### 7.5 Cancel

```
DELETE /v1/jobs/{jobID}
```

Any 2xx. The body is ignored. Cancelling frees the partial upload; the server
should move it to `Failed/` or delete it, and must stop any training in flight.

---

## 8. Watching a job

### 8.1 The stage machine

```
queued -> receiving -> verifying -> training -> exporting -> ready
                                                          \-> failed
   (any stage) -> cancelled
```

| `stage` | Means |
|---|---|
| `queued` | Accepted, waiting behind other work |
| `receiving` | Bytes are arriving |
| `verifying` | Checksums and manifest completeness |
| `training` | Our trainer is running |
| `exporting` | Writing the `.ply` / `.spz` result |
| `ready` | Result is downloadable |
| `failed` | Terminal, with a plain-language `message` |
| `cancelled` | Terminal, by user request |

`ready`, `failed` and `cancelled` are terminal: the client stops listening
after any of them.

### 8.2 Live stream (preferred)

```
GET ws://host:port/v1/jobs/{jobID}/stream
Authorization: Bearer <token>
X-Nimbus-Api-Version: 1
```

The server pushes one JSON message per progress update, as a WebSocket **text**
or **binary** frame (the client accepts either and decodes both as UTF-8 JSON):

```json
{
  "stage": "training",
  "fractionComplete": 0.42,
  "message": "Building the 3D model",
  "timestamp": "2026-09-03T14:19:44Z"
}
```

`fractionComplete` is optional. **Omit it when you genuinely do not know.** The
client draws an indeterminate spinner for a missing fraction and a real bar for
a present one; a fake `0.0` renders as a bar frozen at zero, which reads as
"broken" to a non-technical user. `message` is mandatory and is shown verbatim.

Send an update at least every 10 seconds even when nothing changed, so the
socket does not look dead. Close the socket after the terminal event.

If the socket cannot be established, the client retries after 1, 2, 5, 10 and
20 seconds. If every attempt fails it falls back to polling.

### 8.3 Polling fallback

```
GET /v1/jobs/{jobID}
```

```json
{
  "jobID": "7f3a…",
  "stage": "ready",
  "fractionComplete": 1.0,
  "message": "Finished.",
  "resultManifest": { "scanID": "scan_20260903_141150", "files": [ … ] }
}
```

Polled every 3 seconds until terminal. `fractionComplete` and `resultManifest`
are optional; `resultManifest` should be present once `stage` is `ready`.

A server that implements only this endpoint and no WebSocket still works: the
client falls back automatically after ~38 seconds of failed socket attempts.
Implementing the socket is strongly preferred - 38 seconds of nothing is a bad
first impression.

---

## 9. Bringing the result back

```
GET /v1/jobs/{jobID}/result/manifest
```

Returns a `Manifest` in exactly the same shape as the upload manifest, `scanID`
included (it is mandatory in that type - a result manifest without it fails to
decode).

Then, per file:

```
GET /v1/jobs/{jobID}/result/files/{relativePath}
Range: bytes=0-1048575
```

* Answer `206 Partial Content` with a `Content-Range` header.
* An open-ended `Range: bytes=2097152-` must also be handled: the client sends
  one when it does not know the remaining length.
* Answering `200 OK` with the whole file is legal. The client treats a `200` as
  "that was everything" and stops requesting ranges for that file, so **never**
  answer `200` to a ranged request unless the body really is the entire file.
* The client verifies each file's whole-file SHA-256 against the manifest after
  download and refuses a mismatch.
* Downloads resume: if a partial file is already on the phone, the next attempt
  starts at its current size.

Results land on the phone in `Documents/<brand>/Scans/<scanID>/model/`,
preserving the manifest's relative paths. Keep result relative paths shallow
and match `docs/DATA_FORMAT.md`'s `model/` layout: `model.ply`, `model.spz`,
`model.json`, `background.bin`, `exposure.bin`.

---

## 10. Status codes

| Code | Meaning to the client | What the user is told |
|---|---|---|
| `200`-`299` | Success | - |
| `401`, `403` | Not paired / token rejected | "This phone is not paired with that Booster yet." |
| `404` | Job unknown | "The Booster no longer recognises this job." |
| `409` | Rejected, body is `{accepted:false, reason}` | `reason`, verbatim |
| `426` | API version mismatch; echo `X-Nimbus-Api-Version` | "…different versions (Booster is on N)." |
| anything else | Generic failure | "The Booster returned an unexpected error (HTTP N)." |

There is no error envelope beyond `409`'s. For every other failure the client
ignores the body entirely, so put the human-readable explanation in a `409`
when you want the user to read it.

A network failure, a refused connection or a DNS failure surfaces as "Could not
reach that computer… make sure your phone and computer are on the same Wi-Fi
network and the Booster app is open." A timeout surfaces separately.

---

## 11. Versioning

`X-Nimbus-Api-Version` is a single integer as a string. It is bumped **only**
for a breaking change: a removed or renamed field, a changed field type, a new
enum value the old client cannot decode, or changed resume semantics.

Additive optional fields do not bump it.

On a mismatch the server answers `426 Upgrade Required` and includes its own
version in the header. Do not try to negotiate: a wrong-version pair fails
loudly with "update both to the same version", which is a fixable instruction,
instead of half-working.

---

## 12. Server-side storage

The Booster saves under a configurable root, defaulting to the user's
`Documents/<brand>/Scans`:

```
Documents/Nimbus3D/Scans/
  Incoming/<scanID>/      partially or fully uploaded capture bundles
  Completed/<scanID>/     finished jobs, capture bundle + result/
  Failed/<scanID>/        whatever a failed job left behind, kept for diagnosis
```

The uploaded tree under `Incoming/<scanID>/` mirrors the phone's scan folder
exactly, so `docs/DATA_FORMAT.md` describes it unchanged and the trainer opens
it with the same code path as a locally-produced capture.

The result written into `Completed/<scanID>/result/` is what
`/v1/jobs/{id}/result/*` serves.

---

## 13. One deliberate divergence to know about

`ios/Sources/Export/BoosterBundle.swift` can zip an entire scan folder into a
single `.zip`. That is **not** this protocol, and the Booster does not accept a
zip upload.

The decision, recorded here so nobody has to re-litigate it: **the per-file
chunked manifest transfer above is the Booster transport.** A zip cannot be
resumed part-way, has to be fully written to the phone's storage before the
first byte is sent (doubling peak disk use on a multi-gigabyte house scan), and
would have to be unpacked before verification.

The zip keeps a real, different job: giving the user a single file they can
AirDrop, back up, or hand to a PC by hand when there is no Wi-Fi. Its
reconciliation is a rename, from `packageForBooster` to `packageCaptureBundle`
- which `Core/Contracts.swift`'s `SplatExporting.packageCaptureBundle` already
exposes under the better name. See `CONTRACTS.md`.
