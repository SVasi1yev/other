import threading
import socket
import queue
import signal
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import re
import json
import configparser
import logging
import sys
import os

SPLIT_RGX = re.compile(r'[A-Za-zА-Яа-я0-9]+', re.UNICODE)


def split(string):
    words = re.findall(SPLIT_RGX, string)
    return words


def raise_exp(signal_num, frame):
    raise KeyboardInterrupt


class Server:
    END_TOKEN = b'\n'
    ANSWER_TOKEN = b'<answer>'
    SPLIT_TOKEN = '<split>'
    FINISH_TOKEN = '<finish>'

    def worker(self, queue):
        while True:
            addr, data = tuple(queue.get().split(self.SPLIT_TOKEN))
            if data == self.FINISH_TOKEN:
                return
            try:
                html = requests.get(data).text
            except Exception:
                res = 'Error'.encode()
            else:
                text = BeautifulSoup(html, 'lxml').text
                res = defaultdict(int)
                for word in split(text):
                    res[word] += 1
                res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True)[:self.TOP_N])
                res = json.dumps(res).encode()
            sock = socket.socket()
            sock.connect(('localhost', self.PORT))
            sock.sendall(self.ANSWER_TOKEN + addr.encode() + self.SPLIT_TOKEN.encode() + res + self.END_TOKEN)
            sock.close()

    def __init__(self, config_filename):
        self.config = configparser.ConfigParser()
        self.config.read(config_filename)

        self.WORKERS_NUM = int(self.config.get('Workers config', 'workers_num'))
        self.TOP_N = int(self.config.get('Workers config', 'top_n'))
        self.PORT = int(self.config.get('Connection', 'port'))
        self.CONN_NUM = int(self.config.get('Connection', 'connections_number'))

        logs_file_config = self.config.get('Logs', 'file')
        logging.basicConfig(
            stream=logs_file_config if logs_file_config != '' else sys.stdout,
            level=logging.INFO
        )

        self.sock = None
        self.connections = {}

        self.queues = [
            queue.Queue()
            for _ in range(self.WORKERS_NUM)
        ]

        self.workers = [
            threading.Thread(target=self.worker, args=(self.queues[i],))
            for i in range(self.WORKERS_NUM)
        ]

        self.cur_worker = 0
        self.proc_urls_num = 0

        signal.signal(signal.SIGUSR1, raise_exp)

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('localhost', self.PORT))
        self.sock.listen(self.CONN_NUM)

        config_timeout = self.config.get('Connection', 'timeout')

        for w in self.workers:
            w.start()

        logging.info('Server was started (PID: %s)', os.getpid())

        try:
            while True:
                conn, addr = self.sock.accept()
                logging.info('Connected: %s', addr)
                conn.settimeout(int(config_timeout) if config_timeout != '' else None)
                data = b''
                try:
                    while (len(data) < len(self.END_TOKEN)) or (data[-len(self.END_TOKEN):] != self.END_TOKEN):
                        data += conn.recv(1024)
                except socket.timeout:
                    logging.info('Timeout for connection %s', addr)
                    conn.sendall('timeout\n'.encode())
                    conn.close()
                    logging.info('Connection closed: %s', addr)
                else:
                    logging.info('Get: %s', data[:-1].decode())
                    if len(data) > len(self.ANSWER_TOKEN) and data[:len(self.ANSWER_TOKEN)] == self.ANSWER_TOKEN:
                        conn.close()
                        logging.info('Connection closed: %s', addr)
                        data = data[len(self.ANSWER_TOKEN):]
                        addr, answer = tuple(data.decode().split(self.SPLIT_TOKEN))
                        self.connections[addr].sendall(answer.encode())
                        self.connections[addr].close()
                        logging.info('Connection closed: %s', addr)
                        self.connections.pop(addr)
                        self.proc_urls_num += 1
                    else:
                        self.connections[str(addr)] = conn
                        self.queues[self.cur_worker].put(str(addr) + self.SPLIT_TOKEN + data[:-1].decode())
                        logging.info('Send to worker %s', self.cur_worker)
                        self.cur_worker = (self.cur_worker + 1) % int(self.WORKERS_NUM)
        except KeyboardInterrupt as e:
            pass
        finally:
            self.stop()

    def stop(self):
        for q in self.queues:
            q.put('0' + self.SPLIT_TOKEN + self.FINISH_TOKEN)
        for addr in self.connections:
            self.connections[addr].close()
            self.connections.pop(addr)
        self.sock.close()
        for w in self.workers:
            w.join()
        logging.info('Server was stopped')

if __name__ == '__main__':
    server = Server(sys.argv[1])
    server.start()