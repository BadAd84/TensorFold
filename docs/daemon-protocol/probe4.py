import os, socket, struct

PATH = "/run/dispram/dispram.sock"
KEY = "kindlingai_1"

def conn():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET); s.settimeout(5); s.connect(PATH); return s

def say(s, body, want_fd=True):
    s.sendall(body)
    data, anc, flags, addr = s.recvmsg(4096, socket.CMSG_SPACE(4) if want_fd else 0)
    fds = [struct.unpack("i", c[2][:4])[0] for c in anc if c[0] == socket.SOL_SOCKET and c[1] == socket.SCM_RIGHTS]
    return data, fds

s = conn()
print("A. info with the carveout free:", say(s, f'{{"key":"{KEY}","op":"info"}}'.encode(), False)[0])

for ask in (1, 1048576, 2097151, 2097152, 3145728):
    c = conn()
    d, fds = say(c, f'{{"key":"{KEY}","op":"alloc","size":{ask}}}'.encode())
    got = None
    if fds:
        try:
            got = ("lseek", os.lseek(fds[0], 0, os.SEEK_END))
        except OSError as e:
            got = ("lseek-failed", e.errno)
        for fd in fds: os.close(fd)
    print(f"B. ask={ask}: reply={d} fd={fds} fdsize={got}")
    c.close()

print("C. after those closes, info again:", say(conn(), f'{{"key":"{KEY}","op":"info"}}'.encode(), False)[0])

c = conn()
d, fds = say(c, f'{{"key":"{KEY}","op":"alloc","size":4194304}}'.encode())
print("D. hold a 4 MiB claim:", d, "fd", fds)
d2, fds2 = say(conn(), f'{{"key":"{KEY}","op":"info"}}'.encode(), False)
print("   daemon sees:", d2)
c.close()
print("E. after closing the holder, info:", say(conn(), f'{{"key":"{KEY}","op":"info"}}'.encode(), False)[0])
