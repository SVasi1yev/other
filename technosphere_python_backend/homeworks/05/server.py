#!/home/svasilyev/anaconda3/envs/venv/bin/python3
import configparser
import socket
import json
import sys
import logging
from datetime import datetime

import requests
import dicttoxml


class Server:
    def __init__(self, config_filename):
        self.config = configparser.ConfigParser()
        self.config.read(config_filename)
        logs_file_config = self.config.get('Logs', 'file')
        dicttoxml.LOG.setLevel(logging.ERROR)
        logging.basicConfig(
            stream=logs_file_config if logs_file_config != '' else sys.stdout,
            level=logging.INFO
        )

    def make_request(self, id_):
        response = requests.get(
            f'{self.config.get("API", "url")}/{id_}',
            headers={
                'X-API-KEY': self.config.get('API', 'key'),
            }
        )
        return response.text

    def get_response(self, request):
        start_time = datetime.now()
        splited = request.split('_')
        if len(splited) == 2:
            format_, id_ = splited[0], splited[1]
            if not id_.isdigit():
                response = 'Id must be integer: %x' % id_
                logging.info('%(now)s >> %(response)s',
                             {
                                 'now': datetime.now(),
                                 'response': response
                             })
            if format_ == 'json':
                response = self.make_request(id_)
            elif format_ == 'xml':
                response = dicttoxml.dicttoxml(json.loads(self.make_request(id_))).decode('utf8')
            else:
                response = 'Format must be [json|xml]: %s' % format_
                logging.info('%(now)s >> %(response)s',
                             {
                                 'now': datetime.now(),
                                 'response': response
                             })
            end_time = datetime.now()
            logging.info(
                '%(now)s >> Request %(format_)s_%(id_)s; start: %(start)s;'
                ' end: %(end)s; total time: %(total_time)s; total_size: %(total_size)s',
                {
                    'now': datetime.now(),
                    'format_': format_,
                    'id_': id_,
                    'start': start_time,
                    'end': end_time,
                    'total_time': end_time - start_time,
                    'total_size': len(response)
                }
            )
        else:
            response = 'Two parameters are required: <format>_<id>'
            logging.info('%(now)s >> %(response)s',
                         {
                             'now': datetime.now(),
                             'response': response
                         })
        return response

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('localhost', int(self.config.get('Connection', 'port'))))
        config_timeout = self.config.get('Connection', 'timeout')
        # sock.settimeout(int(config_timeout) if config_timeout != '' else None)
        logging.info('Server was started')

        try:
            while True:
                sock.listen(int(self.config.get('Connection', 'connections_number')))

                conn, addr = sock.accept()
                logging.info('Connected: %s', addr)
                conn.settimeout(int(config_timeout) if config_timeout != '' else None)
                data = b''
                try:
                    while (len(data) < 5) or (data[-5:] != b'<eof>'):
                        data += conn.recv(1024)
                except socket.timeout:
                    logging.info('%(now)s >> timeout for connection %(addr)s',
                                 {
                                     'now': datetime.now(),
                                     'addr': addr
                                 })
                    response = f'timeout for connection {addr}'
                else:
                    response = self.get_response(data[:-5].decode('utf8'))
                conn.sendall((response + '<eof>').encode('utf8'))
                conn.close()
                logging.info('Connection closed: %s', addr)
        finally:
            sock.close()
            logging.info('Server was stopped')


if __name__ == '__main__':
    server = Server(sys.argv[1])
    server.start()
