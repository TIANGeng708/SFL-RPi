import requests
import torch
import time
train_count=0
with open('s3.txt', 'wb') as file:
    file.write("-".encode())
while True:
    while True:
        try:
            r1 = requests.get('http://192.168.1.213:8001/t3_client.txt',timeout=5)
            print(r1.content.decode())
        except:
            continue
        if r1.content.decode() == str(train_count):
            try:
                r2 = requests.get('http://192.168.1.213:8001/t3_send1.pt',timeout=5)
            except:
                continue
            with open('dfx3.pt', 'wb') as file:
                file.write(r2.content)
            try:
                r3 = requests.get('http://192.168.1.213:8001/t3_send2.pt',timeout=5)
            except:
                continue
            with open('label3.pt', 'wb') as file:
                file.write(r3.content)


            break
        time.sleep(10)
    with open('s3.txt', 'wb') as file:
        file.write(str(train_count).encode())

    if (train_count+1)%(60*5)==0:#2指的是用户拥有多少个minibatch
        while True:
            try:
                r4 = requests.get('http://192.168.1.213:8001/avg.txt', timeout=5)
            except:
                continue
            if r4.content.decode() == '+':
                try:
                    time.sleep(15)
                    r5=requests.get('http://192.168.1.213:8001/avg.pt',timeout=5)
                except:
                    continue
                with open('avg.pt','wb') as file:
                    file.write(r5.content)
                break
            time.sleep(5)
        time.sleep(10)
        with open('avg.txt', 'wb') as file:
            file.write('+'.encode())
    train_count+=1