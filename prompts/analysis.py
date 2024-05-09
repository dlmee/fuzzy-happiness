import json
from random import random

class Leaf:
    def __init__(self, value):
        self.value = value
        self.promptLog = None
        self.transLog = None

    def dump(self):
        if self.value is None:
            res = {}
        elif type(self.value) == dict:
            res = self.value
        else:
            res = {"VALUE": self.value}
        if self.promptLog is not None:
            res["PROMPTLOG"] = self.promptLog
        if self.transLog is not None:
            res["TRANSLOG"] = self.transLog
        # status = fresh, agree, fixed, failedElsewhere, unsure
        if "ANNOTATION" not in res:
            res["ANNOTATION"] = {"STATUS": "fresh", "NOTES": "", "GOLD": "", "INDEX": "annotation_idx_%s" % (random(),)}
        return res
        
class Analysis:
    def __init__(self):
        self.d = {}

    # Key must be either a string or a tuple of strings
    # value should be None, a string, or a possibly nested dict of strings
    def add(self, key, value=None):
        if type(key) == str:
            self.d[key] = [Leaf(value)]
        else:
            cur = self.d
            for k in key[:-1]:
                cur = cur.setdefault(k, {})
            cur.setdefault(key[-1], []).append(Leaf(value))

    # Sets the corresponding logs of all leaf nodes in this analysis
    # to match the not-None inputs.
    def setLogs(self, promptLog=None, transLog=None, d=None):
        if d is None:
            self.setLogs(promptLog, transLog, self.d)
        elif type(d) == list:
            for leaf in d:
                if promptLog is not None:
                    leaf.promptLog = promptLog
                if transLog is not None:
                    leaf.transLog = transLog
        else:
            for k in d:
                self.setLogs(promptLog, transLog, d[k])

    def _recursiveMerge(self, myD, oD):
        if type(oD) == list:
            myD.extend(oD)
        else:
            for k in oD:                
                if k not in myD:
                    myD[k] = oD[k]
                else:
                    self._recursiveMerge(myD[k], oD[k])

    def merge(self, other):
        self._recursiveMerge(self.d, other.d)
        
    def dump(self, d = None):
        if d == None:
            return self.dump(self.d)
        elif type(d) == list:
            lst = []
            for leaf in d:
                dmp = leaf.dump()
                if dmp != {}:
                    lst.append(dmp)
            res = {"COUNT": len(d)}
            if len(lst) > 0:
                res["INSTANCES"] = lst
            return res
        else:
            count = 0
            res = {}
            for k in d:
                res[k] = self.dump(d[k])
                count += res[k]["COUNT"]
            res["COUNT"] = count
            return res
