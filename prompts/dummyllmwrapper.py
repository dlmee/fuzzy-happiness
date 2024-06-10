import time
import os
import functools
from db_cache_pickle_V1 import Cache

# Dummy class with __call__, multiply, and add methods
class DummyLLM:
    def __init__(self, log=None):
        self.log = log

    def __call__(self, prompt, js=False, badJS=5):
        time.sleep(1)  # Simulate some work
        result = f"Processed prompt: {prompt}, js: {js}, badJS: {badJS}"
        if self.log:
            self.log(result)
        return result
    
    def multiply(self, x, y):
        time.sleep(1)  # Simulate some work
        result = x * y
        if self.log:
            self.log(f"Multiplying {x} and {y} to get {result}")
        return result
    
    def add(self, x, y):
        time.sleep(1)  # Simulate some work
        result = x + y
        if self.log:
            self.log(f"Adding {x} and {y} to get {result}")
        return result

# Initialize the cache
cache = Cache()

# Apply the cache_class_methods decorator programmatically to the DummyLLM class
CachedDummyLLM = cache.cache_class_methods(DummyLLM)

# Instantiate the cached dummy class and call its methods
dummy_llm = CachedDummyLLM()

# Test __call__ method
print(dummy_llm("Test prompt"))  # This will calculate and store the result
print(dummy_llm("Test prompt"))  # This should retrieve the result from the cache

# Test multiply method
print(dummy_llm.multiply(2, 3))  # This will calculate and store the result
print(dummy_llm.multiply(2, 3))  # This should retrieve the result from the cache

# Test add method
print(dummy_llm.add(5, 7))  # This will calculate and store the result
print(dummy_llm.add(5, 7))  # This should retrieve the result from the cache
