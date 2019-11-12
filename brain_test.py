#%%
import json
import websocket
import ssl

url = "wss://emotivcortex.com:54321/"
#url = "wss://localhost:54321/"

ws = websocket.create_connection(url, sslopt={"cert_reqs": ssl.CERT_NONE})
#"""
auth_request = {
    "jsonrpc": "2.0",
    "method": "authorize",
    "params": {
        "client_id": "KMpwNsdu961HDsNHHrlu5xcoIkXhiJekPvkpHClb",
        "client_secret": "3pXZcl0uj2ETe7BkMdZGWBS3Qfh7EsRNHabbyusdsNdLBsgp3yA1Brp9atsjkvfPN7aY7D88OW4pOsdwA6VP6aO6BV6NJTvjSucv5D64tEwAvWrUR6TVgylLoWkkwKbm",
        "license": "303ad050-31b9-4561-a8b7-20b197b9a467",
        "debit": 1
    },
    "id": 1
}

#"""
"""
auth_request = {
    "jsonrpc": "2.0",
    "method": "hello",
    "params": {
        "hello": "world"
    },
    "id": 1
}
"""
ws.send(json.dumps(auth_request))
response = json.loads(ws.recv())
print(response)


#%%
import socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("localhost",54321))