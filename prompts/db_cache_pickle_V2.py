import sqlite3
import pickle
import functools
import signal
import sys
import os
import uuid
import time
import json

class Cache:
    def __init__(self):
        self.cache_dir = "cache"
        self.models_dir = os.path.join(self.cache_dir, "models")
        self._ensure_directories()
        self.db_path = self._get_db_path()
        signal.signal(signal.SIGINT, self._handle_sigint)
    
    def _ensure_directories(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def _get_db_path(self):
        project_name = os.path.basename(os.getcwd())
        db_name = f"{project_name}_cache.db"
        return os.path.join(self.cache_dir, db_name)
    
    def _create_table_if_not_exists(self, table_name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                id TEXT PRIMARY KEY,
                parameters TEXT,
                output TEXT
            )
            """)
            conn.commit()

        # Create order of entry table if it doesn't exist
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entry_order (
                    table_name TEXT,
                    entry_id TEXT,
                    size INTEGER,
                    timestamp INTEGER,
                    PRIMARY KEY (table_name, entry_id)
                )
            """)
            conn.commit()
    
    def _get_cache(self, table_name, params):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT output FROM "{table_name}" WHERE parameters = ?', (params,))
            row = cursor.fetchone()
            if row:
                output = json.loads(row[0])
                if isinstance(output, dict):
                    items = output.get('_payload', {})
                    if isinstance(items, dict) and items.get("is_model"):
                        model_path = items["model_path"]
                        with open(model_path['_payload'], 'r') as model_file:
                            output = self._deserialize_object(json.load(model_file))
                return self._deserialize_object(output)
        return None
    
    def _set_cache(self, table_name, params, output):
        entry_id = str(uuid.uuid4())  # Generate a UUID for the entry
        if isinstance(output, dict) and output.get("is_model"):
            model_filename = f"model_{table_name}_{hash(params)}.json"
            model_path = os.path.join(self.models_dir, model_filename)
            with open(model_path, 'w') as model_file:
                json.dump(self._serialize_object(output["data"]), model_file)
            output = {"is_model": True, "model_path": model_path}
        
        try:
            serialized_output = json.dumps(self._serialize_object(output), sort_keys=True)
            size = sys.getsizeof(serialized_output)
            timestamp = int(time.time())
            print("About to set a cache in the DB")
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'INSERT OR REPLACE INTO "{table_name}" (id, parameters, output) VALUES (?, ?, ?)', (entry_id, params, serialized_output))
                
                # Insert or replace into order of entry table
                cursor.execute("""
                    INSERT OR REPLACE INTO entry_order (table_name, entry_id, size, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (table_name, entry_id, size, timestamp))
                
                conn.commit()
        except:
            print("Excepted setting cache!")

    def _serialize_object(self, obj):
        if isinstance(obj, dict):
            return {
                "_object": "dict",
                "_payload": {json.dumps(self._serialize_object(k)): self._serialize_object(v) for k, v in obj.items()}
            }
        elif isinstance(obj, list):
            return {
                "_object": "list",
                "_payload": [self._serialize_object(item) for item in obj]
            }
        elif isinstance(obj, tuple):
            return {
                "_object": "tuple",
                "_payload": [self._serialize_object(item) for item in obj]
            }
        elif isinstance(obj, set):
            return {
                "_object": "set",
                "_payload": [self._serialize_object(item) for item in obj]
            }
        elif isinstance(obj, bytes):
            return {
                "_object": "bytes",
                "_payload": obj.decode('utf-8')
            }
        elif isinstance(obj, complex):
            return {
                "_object": "complex",
                "real": obj.real,
                "imag": obj.imag
            }
        elif isinstance(obj, (int, float, str, bool)):
            return {"_object": type(obj).__name__, "_payload": obj}
        elif obj is None:
            return {"_object": "NoneType", "_payload": None}
        elif hasattr(obj, '__dict__'):
            class_name = obj.__class__.__name__
            class_module = obj.__class__.__module__

            self._register_class(class_name, class_module)

            return {
                "_object": "class",
                "_name": class_name,
                "_payload": self._serialize_object(obj.__dict__),
                "module": class_module,
            }
        else:
            try:
                return {"_object": "pickle", "_payload": pickle.dumps(obj).hex()}
            except (TypeError, pickle.PicklingError):
                print(f"Cannot serialize object: {obj}")
                raise


    def _deserialize_object(self, data):
        if isinstance(data, dict):
            obj_type = data.get("_object")
            if obj_type == "class":
                class_name = data["_name"]
                fields = self._deserialize_object(data["_payload"])

                class_module = data.get("module")
                if class_module:
                    module = __import__(class_module)
                    cls = getattr(module, class_name)
                    instance = cls.__new__(cls)
                    instance.__dict__.update(fields)
                    return instance
                else:
                    raise ImportError(f"Class {class_name} not found in registry")
            elif obj_type == "tuple":
                return tuple(self._deserialize_object(item) for item in data["_payload"])
            elif obj_type == "set":
                return set(self._deserialize_object(item) for item in data["_payload"])
            elif obj_type == "list":
                return [self._deserialize_object(item) for item in data["_payload"]]
            elif obj_type == "bytes":
                return bytes(data["_payload"], 'utf-8')
            elif obj_type == "complex":
                return complex(data["real"], data["imag"])
            elif obj_type == "dict":
                deserialized_dict = {}
                for k, v in data["_payload"].items():
                    deserialized_key = self._deserialize_object(json.loads(k))
                    deserialized_value = self._deserialize_object(v)
                    deserialized_dict[deserialized_key] = deserialized_value
                return deserialized_dict
            elif obj_type in {"int", "float", "str", "bool"}:
                return data["_payload"]
            elif obj_type == "NoneType":
                return None
            elif obj_type == "pickle":
                return pickle.loads(bytes.fromhex(data["_payload"]))
            else:
                raise TypeError(f"Unknown type: {obj_type}")
        else:
            return data



    def _register_class(self, class_name, class_module):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS class_registry (
                    class_name TEXT PRIMARY KEY,
                    class_module TEXT
                )
            """)
            cursor.execute("""
                INSERT OR IGNORE INTO class_registry (class_name, class_module)
                VALUES (?, ?)
            """, (class_name, class_module))
            conn.commit()

    def _get_class_module(self, class_name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT class_module FROM class_registry WHERE class_name = ?", (class_name,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return None

    def _serialize_params(self, args, kwargs):
        try:
            if args and hasattr(args[0], '__class__'):
                # Exclude the first argument if it's `self`
                serialized_args = [self._serialize_object(arg) for arg in args[1:]]
            else:
                serialized_args = [self._serialize_object(arg) for arg in args]
            
            serialized_kwargs = {key: self._serialize_object(value) for key, value in kwargs.items()}
            return json.dumps({"args": serialized_args, "kwargs": serialized_kwargs}, sort_keys=True)
        except Exception as e:
            print(f"Failed to serialize parameters: {e}")
            return None

    
    def _deserialize_params(self, params):
        data = json.loads(params)
        args = [self._deserialize_object(arg) for arg in data["args"]]
        kwargs = {key: self._deserialize_object(value) for key, value in data["kwargs"].items()}
        return args, kwargs

    def cached_function(self, func=None, threshold_time=0.005, cls_name=None):
        if func is None:
            def decorator(inner_func):
                return self.cached_function(inner_func, threshold_time, cls_name)
            return decorator

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            overwrite = kwargs.pop('overwrite', False)
            # Use class name in table name if provided
            table_name = f"{cls_name}_{func.__name__}" if cls_name else func.__name__
            params = self._serialize_params(args, kwargs)
            if params:
                self._create_table_if_not_exists(table_name)
                
                if not overwrite:
                    cached_output = self._get_cache(table_name, params)
                    if cached_output is not None:
                        return cached_output

            output = func(*args, **kwargs)
            execution_time = time.time() - start_time

            if threshold_time == 0 or execution_time > threshold_time:
                try:
                    self._set_cache(table_name, params, output)
                except Exception as e:
                    print(f"Failed to cache due to: {e}")

            return output

        return wrapper


    
    def cached_model(self, func, threshold_time=0.005):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            overwrite = kwargs.pop('overwrite', False)
            table_name = func.__name__
            params = self._serialize_params(args, kwargs)
            
            self._create_table_if_not_exists(table_name)
            
            if not overwrite:
                cached_output = self._get_cache(table_name, params)
                if cached_output is not None:
                    return cached_output

            output = func(*args, **kwargs)
            execution_time = time.time() - start_time

            model_output = {"data": self._serialize_object(output), "is_model": True}
            if threshold_time == 0 or execution_time > threshold_time:
                self._set_cache(table_name, params, model_output)
            return output
        return wrapper
    
    def cache_class_methods(self, cls, threshold_time=0.005):
        cls_name = cls.__name__
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value) and not attr_name.startswith('_'):
                wrapped_method = self.cached_function(attr_value, threshold_time, cls_name)
                setattr(cls, attr_name, wrapped_method)
        # Special handling for __call__
        if hasattr(cls, '__call__'):
            setattr(cls, '__call__', self.cached_function(cls.__call__, threshold_time, cls_name))
        return cls


    
    def _handle_sigint(self, signum, frame):
        print("SIGINT received, exiting...")
        sys.exit(0)

    def shrink_cache(self, target_size):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get the current total size
            cursor.execute("SELECT SUM(size) FROM entry_order")
            total_size = cursor.fetchone()[0] or 0
            
            while total_size > target_size:
                # Get the oldest entry
                cursor.execute("SELECT table_name, entry_id, size FROM entry_order ORDER BY timestamp ASC LIMIT 1")
                oldest_entry = cursor.fetchone()
                if not oldest_entry:
                    break
                
                table_name, entry_id, entry_size = oldest_entry[0], oldest_entry[1], oldest_entry[2]
                
                # Remove the oldest entry
                cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (entry_id,))
                cursor.execute("DELETE FROM entry_order WHERE table_name = ? AND entry_id = ?", (table_name, entry_id))
                
                total_size -= entry_size
            
            conn.commit()
