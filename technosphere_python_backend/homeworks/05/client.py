#!/home/svasilyev/anaconda3/envs/venv/bin/python3
import sys
import socket


sock = socket.socket()
try:
    sock.settimeout(20)
    sock.connect(('localhost', int(sys.argv[2])))
    sock.sendall(f'{sys.argv[1]}_{sys.argv[3]}<eof>'.encode('utf8'))

    data = b''
    while (len(data) < 5) or (data[-5:] != b'<eof>'):
        data += sock.recv(1024)

    print(data.decode('utf8')[:-5])
finally:
    sock.close()
