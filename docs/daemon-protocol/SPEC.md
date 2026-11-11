# Lending-daemon protocol: the specification a client is written from

This is the ONLY source an implementation should use. It states what the daemon does, as observed by running it and
talking to its socket on 2026-11-10 (kindling spark-os 0.9.5 on a GB10, driver 580.178.04), plus what is published in
Kindling's own README. It deliberately says nothing about how any existing client is written, and an implementer must
not read `python/dispram.py`, `dispramd.py`, or any other implementation, in any language.

## 1. Transport

- A Unix domain socket, path given by the operator (`/run/dispram/dispram.sock` is the usual one; another daemon may
  listen elsewhere).
- Socket type MUST be `SOCK_SEQPACKET`. `SOCK_STREAM` and `SOCK_DGRAM` are refused at connect time with
  `EINVAL` ("Protocol wrong type for socket").
- One request per datagram, one reply per datagram. Both are UTF-8 JSON objects. Byte order and padding are not
  involved: the payload is text, and the only binary part is the file descriptor that travels in the ancillary data.
- Requests may be pipelined on one connection: an `info` followed by an `alloc` on the same socket works, and the
  connection stays usable afterwards.

## 2. The key

Every request must carry `"key"`, a string. The daemon's key is a per-installation constant; the operator supplies it
(`kindlingai_1` on the reference install). A request with no key, an unknown key, or a payload that is not JSON at all
gets exactly:

    {"ok": false, "error": "missing or unknown key"}

So a key error is not distinguishable from a malformed message from the client's side; treat both as "refused".

## 3. `info`

Request:

    {"key": "<key>", "op": "info"}

Reply, always `ok:true`-less (it carries no `ok` field at all), four integers in bytes:

    {"base": 10739515392, "size": 2145386496, "free": 2145386496, "largest": 2145386496}

- `base`: the reservation's base address (the daemon's own view; not the slice offset).
- `size`: the whole reservation.
- `free`: bytes currently unclaimed.
- `largest`: the largest single claim that would succeed right now (equal to `free` when the reservation is empty,
  and it drops by exactly the granted size while a claim is held).

Numbers are printed with `", "` separators as shown, but a client MUST parse JSON rather than the exact spacing.

## 4. `alloc`

Request:

    {"key": "<key>", "op": "alloc", "size": <integer bytes>}

Success: the reply carries `ok:true`, the reservation `base`, and `size` - the GRANTED size - and exactly one file
descriptor arrives in the ancillary data as `SCM_RIGHTS`:

    {"ok": true, "base": 10739515392, "size": 2097152}

**The daemon rounds the request UP to its 2 MiB granularity.** Observed: request 1 → granted 2097152; 1048576 →
2097152; 2097151 → 2097152; 2097152 → 2097152; 3145728 → 4194304. A client that wants exactly N bytes must ask for a
value that is already a multiple of 2 MiB, and must read the granted `size` from the reply rather than assuming its
request was honoured.

Refusals, with `ok:false`, an `error` string, and NO descriptor:

| Condition | Reply |
|---|---|
| request larger than the whole reservation | `{"ok": false, "error": "size must be an integer from 1 to 2145386496"}` |
| more than `free`/`largest` | `{"ok": false, "error": "not enough free carveout", "base": ..., "size": ..., "free": ..., "largest": ...}` |
| `size` missing, zero, negative, non-integer | `{"ok": false, "error": "size must be an integer from 1 to <reservation>}` |
| unknown `op` (e.g. `"free"`) | `{"ok": false, "error": "unknown op"}` |
| missing/unknown `key` | `{"ok": false, "error": "missing or unknown key"}` |

There is NO op that releases a slice: `free` is an unknown op. See section 6.

## 5. The lent file descriptor

- It is the memory, shared as a POSIX file descriptor. The client maps it with the CUDA virtual-memory API:
  `cuMemAddressReserve` for a range of its own, `cuMemImportFromShareableHandle` with handle type
  `CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR` (1), `cuMemMap`, then `cuMemSetAccess` read-write for the current device.
- **The descriptor is not seekable**: `lseek(fd, 0, SEEK_END)` fails with ESPIPE (errno 29). A client MUST NOT use the
  fd's own length as its size; the granted `size` in the reply is the authority, and the mapping must not exceed it.
- Once imported, the mapping holds its own reference to the memory, so the client may close the descriptor afterwards.

## 6. Lifetime and release

- A claim lives while the connection that asked for it stays open. Observed: with a 4 MiB claim held, `info` on
  another connection reports `free` reduced by exactly 4194304; after the holding connection is closed, `free` returns
  to the full reservation. So: **closing the window socket is the release**, and a client must keep that socket open
  for as long as it wants the memory.
- Unmapping the CUDA range releases the client's reference; closing the socket gives the memory back to the daemon.
  Both are needed for a clean release, in that order.

## 7. What the client must refuse rather than fit

- A daemon that lends less than the ask (only possible if a client asks for a size the daemon then clamps) is a
  refusal, not a smaller pool: the placement above depends on the size it planned for.
- A missing socket, a refused connect, a refused claim, a reply that is not the expected JSON, a reply with no file
  descriptor, and a `size` in the reply smaller than the ask are all load-stopping refusals with the cause named.
- Never map more than the granted size, and never report a carveout larger than what the daemon granted.

## 8. Values for the reference install

Socket `/run/dispram/dispram.sock`, key `kindlingai_1`, reservation `2145386496` bytes (2046 MiB) at base
`10739515392`, granularity 2 MiB. Another daemon that speaks this protocol may differ in all three; the client takes
the socket and the key from settings and the sizes from the daemon's replies.
