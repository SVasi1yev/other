import socket
import sys
import threading

urls_file = sys.argv[1]
workers_num = int(sys.argv[2])

urls = []
with open('urls.txt') as f:
    for line in f:
        urls.append(line)


def make_requests(worker_num, urls):
    for url in urls:
        sock = socket.socket()
        sock.connect(('localhost', 10001))
        sock.sendall(url.encode())

        data = b''
        while (len(data) < 1) or (data[-1:] != b'\n'):
            data += sock.recv(1024)

        print(str(worker_num) + ' >> ' + data.decode())
        sock.close()


workers = [
    threading.Thread(target=make_requests, args=(i, urls[i::workers_num],))
    for i in range(workers_num)
]

for w in workers:
    w.start()
for w in workers:
    w.join()
