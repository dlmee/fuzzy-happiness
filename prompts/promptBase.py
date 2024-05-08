import sys
sys.path.append("../")
from llmWrapper import LLMWrapper
from abc import ABC, abstractmethod
from multiprocessing import get_context
import hashlib
import os
from datetime import datetime
import json
from glob import glob
import importlib
from prompts.analysis import Analysis
from xriLog import XRILog

def testFunc(x):
    clsName, useJs, use = x
    llm = LLMWrapper(js = useJs, log=print)
    modName = "prompts." + clsName[0].lower() + clsName[1:]
    module = importlib.import_module(modName)
    class_ = getattr(module, clsName)
    instance = class_(llm)
    res = []
    for d in use:
        if "ANNOTATION" in d:
            r = instance.golden(d)
            if r is not None:
                res.append(r)
        else:
            res.append(instance.test(d))
    return res


class PromptBase(ABC):
    @property
    @abstractmethod
    def prompt(self) -> str:
        pass
    
    returnJS = True
    
    def __init__(self, llm, log=None):
        self.llm = llm
        if log == None:
            self.log = self.llm.log
        else:
            self.log = log
        if type(self.log) == XRILog:
            genPath = self.log.genPath
        else:
            genPath = "generated"
        self.logPath = f"{genPath}/prompts/logs/{self.__class__.__name__}"
        self.js = True
        self.lastLog = None

    def _toLog(self, pr, res, fields, logFoo):
        js = {
            "LLMTYPE": self.llm.type,
            "TIME": f"{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "PROMPT": self.prompt,
            "FIELDS": fields,
            "INPUT": pr,
            "OUTPUT": res}
        if type(self.log) == XRILog:
            js["TRANSLOG"] = self.log.fileName
        path = "/".join(logFoo.split("/")[:-1])
        os.makedirs(path, exist_ok=True)
        with open(logFoo, "w", encoding="utf-8") as fo:
            json.dump(js, fo, indent=2, ensure_ascii = False)

    def readLog(self, logFoo):
        with open(logFoo) as fi:
            data = fi.read()
        return json.loads(data)
            
    def _fromLog(self, pr, logFoo):
        if os.path.isfile(logFoo):
            try:
                d = self.readLog(logFoo)
            except:
                return None
            if d["INPUT"] == pr and d["LLMTYPE"] == self.llm.type:
                return d["OUTPUT"]
        return None

    def valid(self, res):
        return True

    def format_prompt_kwargs(self, kwargs: dict):
        return kwargs

    def __call__(self, **kwargs) -> dict:
        """
        Returns a Python dictionary representing the JSON output of the LLM call
        """
        formatted_kwargs = self.format_prompt_kwargs(kwargs)
        pr = self.prompt.format(**formatted_kwargs)
        hash = hashlib.md5((pr + self.llm.type).encode()).hexdigest()
        logFoo = "%s/%s.log" % (self.logPath, hash)
        self.lastLog = logFoo
        if "rerun" in kwargs and kwargs["rerun"]:
            res = None
        else:
            res = self._fromLog(pr, logFoo)

        if not self.valid(res):
            res = None
        if res != None:
            self.llm.logPrompt(pr)
            self.llm.logResponse(res)
            return res
        tail = ""
        for retries in range(3):
            res = self.llm(pr + tail, js=self.returnJS)
            if self.valid(res):
                break
            if self.returnJS:
                tail = json.dumps(res, indent=2)
                tail += "\nYour reponse does not match the specified format.  Please correct it."
        else:
            raise ValueError("The LLM's response failed the validator after all retries")
        self._toLog(pr, res, kwargs, logFoo)
        return res

    def test(self, d):
        """
        The input dict contains INPUT and OUTPUT and some other stuff.
        INPUT is what the llm received, OUTPUT is what it returned, and PROMPTLOG is the name of the log file
        containing this info.  The log file is there to be copied into the return value where it makes sense.
        This should run the test prompt(s) on the given dictionary and return a pair of Analysis objects.
        The first Analysis contains structured information about successful aspects of the test,
        and the second contains structured information about failures.
        If the test fails, it should return None.
        """
        raise NotImplementedError("You must implement the 'test' method to test log files")

    def goldenKey(self, same):
        if same:
            return "GOLD"
        else:
            return "PYRITE"
    
    def golden(self, d):
        if not hasattr(self, "fungible"):
            from prompts.fungiblePrompt import FungiblePrompt
            self.fungible = FungiblePrompt(self.llm, self.log)
        promptLog = self.readLog(d["PROMPTLOG"])
        value = {}
        for k in d:
            if k not in ["PROMPTLOG", "TRANSLOG"]:
                value[k] = d[k]
        good, bad, uncat = Analysis(), Analysis(), Analysis()
        if d["ANNOTATION"]["STATUS"] == "agree":
            response1 = json.dumps(promptLog["OUTPUT"], indent=2)
        elif d["ANNOTATION"]["STATUS"] == "fixed":
            response1 = d["ANNOTATION"]["GOLD"]
        else:
            uncat.add(d["ANNOTATION"]["STATUS"].upper(), value)
            return good, bad, uncat
        res = self.__call__(**promptLog["FIELDS"])
        response2 = json.dumps(res, indent=2)
        same = self.fungible(prompt=promptLog["INPUT"], response1=response1, response2=response2)
        key = self.goldenKey(same)
        value["FUNGLOG"] = self.fungible.lastLog
        if same:
            good.add(key, value)
        else:
            bad.add(key, value)
        return good, bad, uncat
    
    def _testAll(self, use, threads=0):
        """ threads == 0 --> Run one thread for each item in 'use'
            threads == 1 --> Run everything in the main thread
            threads > 1  --> Use that many threads """
        res = []
        if threads == 1:
            for js in use:
                res.append(self.test(js))
        else:
            if threads == 0 or threads > len(use):
                threads = len(use)
            cls = self.__class__.__name__
            useList = []
            for i in range(threads):
                useList.append((cls, self.llm.defaultJS, use[i*len(use)//threads:(i+1)*len(use)//threads]))
            with get_context("spawn").Pool(processes = threads) as pool:
                results = pool.map(testFunc, useList)
            res = []
            for r in results:
                res.extend(r)
        goods, bads, uncats = Analysis(), Analysis(), Analysis()
        for d, goodBad in zip(use, res):
            if goodBad is None:
                continue
            if len(goodBad) == 2:
                good, bad = goodBad
                uncat = Analysis()
            else:
                good, bad, uncat = goodBad
                
            if "PROMPTLOG" in d:
                good.setLogs(promptLog=d["PROMPTLOG"])
                bad.setLogs(promptLog=d["PROMPTLOG"])
                uncat.setLogs(promptLog=d["PROMPTLOG"])
            if "TRANSLOG" in d:
                good.setLogs(transLog=d["TRANSLOG"])
                bad.setLogs(transLog=d["TRANSLOG"])
                uncat.setLogs(transLog=d["TRANSLOG"])
            goods.merge(good)
            bads.merge(bad)
            uncats.merge(uncat)
        self.log("There were %s prompts analyzed.  NOTE: one prompt does not necessarily produce one result." % (len(res),))
        return goods, bads, uncats

    def _prettyPrint(self, d, tabs=""):
        if type(d) == dict:
            for k in d:
                if k not in ["COUNT", "INSTANCES", "ANNOTATION"]:
                    self.log("%s%s: %s" % (tabs, k, d[k]["COUNT"]))
                    self._prettyPrint(d[k], tabs+"  ")
    
    def _showResults(self, goods, bads, uncats):
        goodD = goods.dump()
        badD = bads.dump()
        uncatD = uncats.dump()
        res = {"GOOD": goodD, "BAD": badD}
        if len(uncatD) > 1:
            res["UNCATEGORIZED"] = uncatD
        self._prettyPrint(res)
        self.log("Summary: +%s / -%s" % (goodD["COUNT"], badD["COUNT"]))
        self.llm.showStats()
        return res
    
    def _before(self, bef, aft):
        return int(bef.replace("_", "")) < int(aft.replace("_", ""))

    def getUsable(self, startTime, endTime, recent):
        allLogs = glob(self.logPath + "/*.log")
        use = []
        for foo in allLogs:
            try:
                with open(foo) as fi:
                    d = json.loads(fi.read())
                if "PROMPT" not in d or d["PROMPT"] != self.prompt:
                    continue
                if startTime is not None and self._before(d["TIME"], startTime):
                    continue
                if endTime is not None and not self._before(d["TIME"], endTime):
                    continue
                d["PROMPTLOG"] = foo
                use.append(d)
            except:
                pass
        if len(use) == 0:
            raise Exception("No matching log files!")
        if recent > 0:
            times = {}
            for d in use:
                t = int(d["TIME"].replace("_", ""))
                if t not in times:
                    times[t] = []
                times[t].append(d)
            s = sorted(list(times.keys()), reverse=True)
            newUse = []
            for t in s:
                for d in times[t]:
                    newUse.append(d)
                    if len(newUse) == recent:
                        break
                if len(newUse) == recent:
                    break
            use = newUse
        return use

            
    def testDates(self, startTime=None, endTime=None, recent=0, threads=0):
        use = self.getUsable(startTime, endTime, recent)
        goods, bads, uncats = self._testAll(use, threads)
        return self._showResults(goods, bads, uncats)

    def genFresh(self, startTime=None, endTime=None, recent=0):
        use = self.getUsable(startTime, endTime, recent)
        full = Analysis()
        for d in use:
            a = Analysis()
            a.add("full")
            if "PROMPTLOG" in d:
                a.setLogs(promptLog=d["PROMPTLOG"])
            if "TRANSLOG" in d:
                a.setLogs(transLog=d["TRANSLOG"])
            full.merge(a)
        dump = full.dump()
        self._prettyPrint(dump)
        self.llm.showStats()
        return dump
    
    def _getInstances(self, d):
        if type(d) == list:
            filtered = []
            for i in d:
                #if i["ANNOTATION"]["STATUS"] in ["agree", "fixed"]:
                    filtered.append(i)
            return filtered
        lst = []
        for k in d:
            if k != "COUNT":
                lst.extend(self._getInstances(d[k]))
        return lst
    
    def testGolden(self, foo, limit=0, threads=0):
        with open(foo) as fi:
            d = json.loads(fi.read())
            instances = self._getInstances(d)
            if limit > 0:
                instances = instances[:limit]
        if len(instances) == 0:
            raise ValueError("The golden set has no data!")
        goods, bads, uncats = self._testAll(instances, threads)
        return self._showResults(goods, bads, uncats)
