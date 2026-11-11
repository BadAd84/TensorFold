# The display-memory daemon protocol, observed

What the daemon does, established by probing the instance already running on a GB10 (kindling spark-os 0.9.5,
driver 580.178.04), plus what Kindling publishes in its own README. Nothing here describes an implementation: the
four probe scripts beside this file are the whole of the evidence, and anyone can re-run them and get the same
answers.

## Method

Each script connects to `/run/dispram/dispram.sock` and speaks to the daemon directly:

| Script | What it asks |
|---|---|
| `probe.py` | Which socket type the daemon accepts, and what a keyless request gets |
| `probe2.py` | A real claim: the reply's fields, the descriptor, and `lseek` on it |
| `probe3.py` | The request vocabulary and the refusal set, while the reservation was held by a running serve |
| `probe4.py` | The success path with the reservation free: rounding, accounting, release |

## Transport

- `AF_UNIX` with `SOCK_SEQPACKET` connects. `SOCK_STREAM` and `SOCK_DGRAM` are refused at connect with `EINVAL`
  ("Protocol wrong type for socket").
- One JSON object per datagram, one reply per datagram. An `info` followed by an `alloc` on the same connection
  works, and the connection stays usable.
- The key is a per-installation constant, published in Kindling's README line 32: `"key": "kindlingai_1"`.

## What the daemon says, as observed

| Probe | Observed |
|---|---|
| empty object, or no key | `{"ok": false, "error": "missing or unknown key"}` |
| `info` with the key, nothing held | `{"base": 10739515392, "size": 2145386496, "free": 2145386496, "largest": 2145386496}` |
| `alloc` size 1 | `{"ok": true, "base": 10739515392, "size": 2097152}` plus one fd by `SCM_RIGHTS` |
| `alloc` 1048576, 2097151, 2097152 | granted **2097152** in each case |
| `alloc` 3145728 | granted **4194304** |
| `alloc` 2145386497 | `{"ok": false, "error": "size must be an integer from 1 to 2145386496"}` |
| `alloc` with the reservation held | `{"ok": false, "error": "not enough free carveout", "base": ..., "size": ..., "free": ..., "largest": ...}` |
| `op: "free"`, and a body with no op | `{"ok": false, "error": "unknown op"}` |
| `lseek(fd, 0, SEEK_END)` on the lent fd | `OSError: [Errno 29] Illegal seek` |
| a 4 MiB claim held, `info` on another connection | `free` and `largest` each reduced by exactly 4194304 |
| the holding connection closed | `free` back to 2145386496 |

## What those observations decide for a client

1. **The reply's `size` is the only size authority.** The lent descriptor is not seekable, so its length cannot be
   read; a client maps the granted size and nothing more.
2. **The daemon rounds an ask UP** to its 2 MiB granularity. A client that wants at least N must ask a whole number
   of slices, rounding up, or the daemon's round-up can turn a size that would have fitted into a refusal - and a
   client that rounded its ask down would be granted less than it planned for. Either way a grant below the ask is a
   refusal, never a smaller pool.
3. **`ok` must be true, not merely a size present.** The "not enough free carveout" refusal carries `base`, `size`,
   `free` and `largest`, so a client that checked only the size would take a refusal as a grant.
4. **Closing the connection is the release.** There is no op for it, and the daemon's accounting returns to the full
   reservation when the window socket closes.

## Attribution

Protocol facts above: observed on 2026-11-10 with the four scripts in this directory, run against the running
daemon. No kindling client source was used to establish them. The wire constants - the socket path and the key - are
the daemon's own configuration, and its README publishes both.
