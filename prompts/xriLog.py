from datetime import datetime
from random import random
import os
import atexit

class XRILog:
    def __init__(self, genPath=".", logPath="logs", fileName=None, doPrint = True):
        self.genPath = genPath
        if fileName is None:
            fileName = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{random()}.log'
        self.fileName = f"{genPath}/{logPath}/{fileName}"
        path = "/".join(self.fileName.split("/")[:-1])
        os.makedirs(path, exist_ok = True)
        self.log = open(self.fileName, 'w', encoding='utf-8')
        atexit.register(lambda: self.log.close())
        self.doPrint = doPrint

    def __call__(self, s: str):
        self.log.write(s + '\n')
        self.log.flush()
        if self.doPrint:
            print(s)
