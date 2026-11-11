import json, os, socket, struct, time

PATH = "/run/dispram/dispram.sock"
KEY = "kindlingai_1"

def conn():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    s.settimeout(5)
    s.connect(PATH)
    return s

def say(s, body, want_fd=False):
    s.sendall(body)
    data, anc, flags, addr = s.recvmsg(4096, socket.CMSG_SPACE(4) if want_fd else 0)
    fds = [struct.unpack("i", c[2][:4])[0] for c in anc if c[0] == socket.SOL_SOCKET and c[1] == socket.SCM_RIGHTS]
    return data, fds

print("1. two requests on one connection (info then alloc):")
s = conn()
print("   info :", say(s, f'{{"key":"{KEY}","op":"info"}}'.encode())[0])
print("   alloc:", say(s, f'{{"key":"{KEY}","op":"alloc","size":2097152}}'.encode(), True))
s.close()

print("\n2. key on every message? (info ok, then a keyless alloc on the same connection):")
s = conn()
print("   info  :", say(s, f'{{"key":"{KEY}","op":"info"}}'.encode())[0])
try:
    print("   alloc :", say(s, '{"op":"alloc","size":2097152}'.encode(), True))
except Exception as e:
    print("   alloc :", type(e).__name__, e)
s.close()

print("\n3. size handling (largest={}), asks of 1, 2MiB-1, 3MiB:".format(2145386496))
for ask in (1, 2097151, 3145728):
    s = conn()
    d, fds = say(s, f'{{"key":"{KEY}","op":"alloc","size":{ask}}}'.encode(), True)
    print(f"   ask={ask}: reply={d} fds={fds}")
    s.close()

print("\n4. over-ask (largest+1):")
s = conn()
print("   ", say(s, f'{{"key":"{KEY}","op":"alloc","size":2145386497}}'.encode(), True))
s.close()

print("\n5. malformed and unknown-op replies:")
for body in (b'not json', b'{"key":"' + KEY.encode() + b'"}', b'{"key":"' + KEY.encode() + b'","op":"free"}', b'{"key":"wrong","op":"info"}'):
    s = conn()
    print("   ", body, "->", say(s, body)[0])
    s.close()

print("\n6. does closing a connection with a claim free it (alloc, drop socket, alloc the whole reservation):")
s = conn()
d, fds = say(s, f'{{"key":"{KEY}","op":"alloc","size":2145386496}}'.encode(), True)
print("   first claim:", d, "fds:", fds)
for fd in fds: os.close(fd)
s.close()
time.sleep(0.3)
s = conn()
print("   after close:", say(s, f'{{"key":"{KEY}","op":"alloc","size":2145386496}}'.encode(), True))
s.close()

print("\n7. key length / unknown key:")
s = conn()
print("   ", say(s, b'{"key":"x","op":"info"}')[0])
s.close()
