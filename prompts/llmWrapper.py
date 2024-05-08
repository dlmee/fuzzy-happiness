from openai._exceptions import RateLimitError
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage
import tiktoken
from datetime import datetime
from time import time, sleep
import os
from glob import glob
import json
from copy import deepcopy
from random import random

class LLMWrapper:
    def __init__(self, log=None):
        # if log is None, we're ONLY postprocessing existing files, not running new stuff
        if log is not None:
            os.makedirs('calls', exist_ok=True)
            CHAT_MODEL = os.environ["CHAT_MODEL"] if "CHAT_MODEL" in os.environ else "gpt-4"
            if CHAT_MODEL == "gpt-4":
                OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
                if OPENAI_API_KEY[-1] == '\r': OPENAI_API_KEY = os.environ["OPENAI_API_KEY"][:-1]
                self.llm = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY, temperature=0)
            self.encoder = tiktoken.encoding_for_model("gpt-4")
            self.log = log
        self.running = None
            
        start = f'{datetime.now().strftime("%Y%m")}00_000000'
        self.postProc(start, None, "This month")
        thisMonth = self.running["cost"] if self.running is not None and "cost" in self.running else 0

        start = f'{datetime.now().strftime("%Y%m%d")}_000000'
        self.postProc(start, None, "Today")
        today = self.running["cost"] if self.running is not None and "cost" in self.running else 0

        self.running = {"thisMonth": thisMonth, "today": today}

    def _accumulate(self, cur):
        if self.running is None:
            self.running = cur
        else:
            for key in cur:
                if key in self.running:
                    self.running[key] += cur[key]
                else:
                    self.running[key] = cur[key]
        if "thisMonth" in self.running:
            self.running["thisMonth"] += cur["cost"]
        if "today" in self.running:
            self.running["today"] += cur["cost"]

    def reset(self):
        if "running" in self.__dict__:
            res = deepcopy(self.running)
            keys = list(self.running.keys())
            for key in keys:
                if key not in ["thisMonth", "today"]:
                    del self.running[key]
        else:
            self.running = None
            res = None
        return res

    def _before(self, bef, aft):
        b = bef.split("_")
        bYMD, bHMS = b[0], b[1]
        a = aft.split("_")
        aYMD, aHMS = a[0], a[1]
        if int(bYMD) < int(aYMD):
            return True
        if int(bYMD) > int(aYMD):
            return False
        return int(bHMS) < int(aHMS)

    def postProc(self, startTime=None, endTime=None, name="Specified"):
        thisMonth, today = None, None
        if self.running is not None and "thisMonth" in self.running:
            thisMonth = self.running["thisMonth"]
        if self.running is not None and "today" in self.running:
            today = self.running["today"]
        self.running = None
        success = 0
        fail = 0
        for foo in glob("calls/*"):
            fname = foo.split("/")[-1].split(".")[0]
            if (startTime is not None and self._before(fname, startTime)) or (endTime is not None and not self._before(fname, endTime)):
                continue
            with open(foo) as fi:
                txt = fi.read()
                try:
                    cur = json.loads(txt, strict=False)
                    self._accumulate(cur)
                    success += 1
                except Exception as e:
                    import traceback
                    #traceback.print_exc()
                    fail += 1
        print("%s: Succesfully processed %s files, failed on %s files" % (name, success, fail))
        if thisMonth is not None:
            self.running["thisMonth"] = thisMonth
        if today is not None:
            self.running["today"] = today
        return self.running
    
    def __call__(self, prompt, js = False, badJS = 5):
        foo = f'calls/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{random()}.txt'
        with open(foo, 'w') as fo:
            self.log("-"*60)
            self.log("REQUEST:\n%s\n\n" % (prompt,))
            msg = SystemMessage(content = prompt)
            inTokens = len(self.encoder.encode(prompt))
            start = time()
            retry = True
            dur = 0
            while retry:
                start = time()
                try:
                    res = self.llm.invoke([msg])
                    dur += time() - start
                    retry = False
                except RateLimitError as e:
                    dur += time() - start
                    self.log("RATE LIMITED: SLEEPING FOR 10 SECONDS")
                    sleep(10)
            txt = res.content
            outTokens = len(self.encoder.encode(txt))
            inCost = 0.03*inTokens / 1000
            outCost = 0.06*outTokens / 1000
            self.log("-"*60)
            self.log("RESPONSE:\n%s\n\n" % (txt,))
            entry = {"promptLen": len(prompt),
                     "promptTokens": inTokens,
                     "responseLen": len(txt),
                     "responseTokens": outTokens,
                     "duration": dur,
                     "inCost": inCost,
                     "outCost": outCost,
                     "cost": inCost + outCost,
                     }
            self._accumulate(entry)
            dump = json.dumps(entry)
            fo.write(dump)
        if js:
            try:
                return json.loads(txt, strict=False)
            except json.decoder.JSONDecodeError as e:
                if badJS == 0:
                    raise e
                return self.__call__(prompt, True, badJS-1)
        else:
            return txt

if __name__ == "__main__":
    start = "20240408_164500"
    end = None
    llm = LLMWrapper()
    stats = llm.postProc(start, end)
    for key in stats:
        print("%s: %s" % (key, stats[key]))
    
    
