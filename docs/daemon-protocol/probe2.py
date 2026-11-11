import os, socket, struct

PATH = "/run/dispram/dispram.sock"
def call(body, want_fd=False):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.settimeout(5)
    s.connect(PATH)
    s.sendall(body)
    data, anc, flags, addr = s.recvmsg(4096, socket.CMSG_SPACE(4))
    fds = [struct.unpack("i", c[2][:4])[0] for c in anc if c[0] == socket.SOL_SOCKET and c[1] == socket.SCM_RIGHTS]
    return s, data, fds

print("== alloc with the key ==")
s, data, fds = call(b'{"key":"kindlingai_1","op":"alloc","size":2097152}', True)
print("reply:", data)
print("fds:", fds)
if fds:
    fd = fds[0]
    print("fd size (lseek):", os.lseek(fd, 0, os.SEEK_END))
    # the socket closing is the release: the daemon frees the slice when the last reference goes
s.close()
print("closed the window socket; daemon should have the slice back")

print("\n== an oversized ask ==")
s, data, fds = call(b'{"key":"kindlingai_1","op":"alloc","size":4294967296}', True)
print("reply:", data, "fds:", fds); s.close()

print("\n== a malformed body ==")
s, data, fds = call(b'not json', True)
print("reply:", data, "fds:", fds); s.close()

print("\n== an unknown op ==")
s, data, fds = call(b'{"key":"kindlingai_1","op":"free"}', True)
print("reply:", data, "fds:", fds); s.close()
