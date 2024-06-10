import json
from collections import defaultdict
import threading
from dotenv import load_dotenv
from llmWrapper import LLMWrapper
from generic_prompt import GENP
from batched_prompt import BATP
from align_prompt import ALIP
from datetime import datetime
from random import random
from db_cache_pickle_V2 import Cache
import atexit
import os


load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

cache = Cache()

class Multithread_LLM():
    def __init__(self, targets, threads=10, languages = ("Unknown", "English")) -> None:
        self.open_log()
        self.llm = LLMWrapper(self.log)
        #self.llm = DummyLLM()
        self.lock = threading.Lock()
        with open(targets, 'r') as inj:
            mytargets = json.load(inj)
        myresults = self.batch_hub(mytargets, threads=threads, languages=languages)
        with open("testdb_out.json", "w") as outj:
            json.dump(myresults, outj, indent=4, ensure_ascii=False)
        

    def open_log(self):
        os.makedirs('logs', exist_ok=True)
        self.log_foo = f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{random()}.log'
        self.debug_log = open(f'logs/{self.log_foo}', 'w', encoding='utf-8')
        atexit.register(lambda: self.debug_log.close())

    def log(self, s: str):
        self.debug_log.write(s + '\n')
        self.debug_log.flush()
        print(s)

    def batch_hub(self, my_targets, threads: int, languages: tuple):
        allresults = {}
        lang1, lang2 = languages
        template = GENP()
        template = BATP()
        #template = ALIP()
        batch = []
        counter = 0
        bb = []
        for k,v in my_targets.items():
            if k == '**length**': continue
            if len(v['forms']) > 1: continue
            try:
                bb.append(v['forms'][0])
                if len(bb) == 10:
                    #batch.append((template.make_message(lang1, lang2, v['forms']), allresults, k)) This is for unbatched
                    counter += 1
                    batch.append((template.make_message(lang1, lang2, bb), allresults, counter))
                    bb = []
            except KeyError as e:
                print(f"Key error {e}")
            #if counter == 5: break
        print(f"the total number of calls is {counter}")
        if threads > len(batch):
            threads = len(batch)
        batches = [[] for _ in range(threads)]
        counter = 0
        while batch:
            batches[counter].append(batch.pop())
            counter = counter + 1
            if counter == threads:
                counter = 0
        batches = [(b) for b in batches]

        threads = [threading.Thread(target=self.llm_call, args=tuple([b])) for b in batches]

        # Start threads
        for thread in threads:
            thread.start()

        # Wait for threads to finish
        for thread in threads:
            thread.join()
        print("and done!")
        return allresults
    
    def llm_call(self, batch):
        #Restructured the call to now have all the elements for both calls.
        for call, storage, label in batch:
            try:
                response = self.llm(call, js=True)
                with self.lock:
                    storage[label] = response
                print("and done!")
            except json.decoder.JSONDecodeError as e:
                print(f"Failed to process {label} with error {e}")
            except KeyboardInterrupt as e2:
                return




if __name__ == "__main__":
    mythreads = Multithread_LLM('swahili/new_dictionary_afx_final.json', languages=('Swahili', 'English'), threads=1)





