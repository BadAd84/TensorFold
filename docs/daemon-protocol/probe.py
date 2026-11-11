import json, os, socket, struct

PATH = "/run/dispram/dispram.sock"
TYPES = {"SEQPACKET": socket.SOCK_SEQPACKET, "STREAM": socket.SOCK_STREAM, "DGRAM": socket.SOCK_DGRAM}

def try_type(name, typ, payload):
    s = socket.socket(socket.AF_UNIX, typ)
    s.settimeout(3)
    try:
        s.connect(PATH)
    except OSError as e:
        s.close()
        return f"{name}: connect failed: {e}"
    try:
        if typ == socket.SOCK_DGRAM:
            s.send(payload)
        else:
            s.sendall(payload)
        if typ == socket.SOCK_STREAM:
            data = s.recv(4096)
        else:
            data, anc, flags, addr = s.recvmsg(4096, socket.CMSG_SPACE(4))
        fds = [struct.unpack("i", c[2][:4])[0] for c in anc if c[0] == socket.SOL_SOCKET and c[1] == socket.SCM_RIGHTS] if typ != socket.SOCK_STREAM else []
        return f"{name}: reply={data!r} fds={fds}"
    except Exception as e:
        return f"{name}: {type(e).__name__}: {e}"
    finally:
        s.close()

print("== socket type probe, empty object ==")
for name, typ in TYPES.items():
    print(try_type(name, typ, b"{}"))

print("\n== request vocabulary probe on the type that answered ==")
for body in (b'{"op":"info"}', b'{"op":"Info"}', b'{"op":"hello"}', b'{"key":"kindlingai_1","op":"info"}', b'{"op":"alloc","size":2097152}'):
    print(try_type("SEQPACKET", socket.SOCK_SEQPACKET, body))
