#python送信側

import socket
import random
import time

HOST = '127.0.0.1'
PORT = 50007

client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
while True:
    a = random.randrange(3)
    result = str(a)
    print(a)
    client.sendto(result.encode('utf-8'),(HOST,PORT))
    time.sleep(1.0)