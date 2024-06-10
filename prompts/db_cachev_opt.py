import uuid
import time
import sys
import sqlite3
import json
import functools
import signal
import os

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
                    execution_time REAL,
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
                    items = output.get('items', {})
                    if isinstance(items, dict) and items.get("is_model"):
                        model_path = items["model_path"]
                        with open(model_path['value'], 'r') as model_file:
                            output = self._deserialize_object(json.load(model_file))
                return self._deserialize_object(output)
        return None
    
    def _set_cache(self, table_name, params, output, execution_time):
        entry_id = str(uuid.uuid4())  # Generate a UUID for the entry
        if isinstance(output, dict) and output.get("is_model"):
            model_filename = f"model_{table_name}_{hash(params)}.json"
            model_path = os.path.join(self.models_dir, model_filename)
            with open(model_path, 'w') as model_file:
                json.dump(self._serialize_object(output["data"]), model_file)
            output = {"is_model": True, "model_path": model_path}
        
        serialized_output = json.dumps(self._serialize_object(output), sort_keys=True)
        size = sys.getsizeof(serialized_output)
        timestamp = int(time.time())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'INSERT OR REPLACE INTO "{table_name}" (id, parameters, output) VALUES (?, ?, ?)', (entry_id, params, serialized_output))
            
            # Insert or replace into order of entry table
            cursor.execute("""
                INSERT OR REPLACE INTO entry_order (table_name, entry_id, size, timestamp, execution_time)
                VALUES (?, ?, ?, ?, ?)
            """, (table_name, entry_id, size, timestamp, execution_time))
            
            conn.commit()

    def _serialize_object_recursive(self, obj):
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
                "_payload": obj.decode('utf-8')  # Convert bytes to a string
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

            # Register the class in the registry table
            self._register_class(class_name, class_module)

            return {
                "_object": "class",
                "_name": class_name,
                "_payload": self._serialize_object(obj.__dict__),
                "module": class_module,
            }
        else:
            return {"_object": "unknown", "_payload": str(obj)}
        
    def _serialize_object(self, obj):
        stack = [(None, obj)]
        serialized_obj = {}

        while stack:
            parent, current_obj = stack.pop()

            if isinstance(current_obj, dict):
                serialized_dict = {
                    "_object": "dict",
                    "_payload": {}
                }
                for k, v in current_obj.items():
                    serialized_key = json.dumps(self._serialize_object(k))
                    stack.append((serialized_dict["_payload"], (serialized_key, v)))
                if parent is None:
                    serialized_obj = serialized_dict
                else:
                    parent[current_obj[0]] = serialized_dict

            elif isinstance(current_obj, list):
                serialized_list = {
                    "_object": "list",
                    "_payload": []
                }
                for item in current_obj:
                    stack.append((serialized_list["_payload"], item))
                if parent is None:
                    serialized_obj = serialized_list
                else:
                    parent[current_obj[0]] = serialized_list

            elif isinstance(current_obj, tuple):
                serialized_tuple = {
                    "_object": "tuple",
                    "_payload": []
                }
                for item in current_obj:
                    stack.append((serialized_tuple["_payload"], item))
                if parent is None:
                    serialized_obj = serialized_tuple
                else:
                    parent[current_obj[0]] = serialized_tuple

            elif isinstance(current_obj, set):
                serialized_set = {
                    "_object": "set",
                    "_payload": []
                }
                for item in current_obj:
                    stack.append((serialized_set["_payload"], item))
                if parent is None:
                    serialized_obj = serialized_set
                else:
                    parent[current_obj[0]] = serialized_set

            elif isinstance(current_obj, bytes):
                serialized_bytes = {
                    "_object": "bytes",
                    "_payload": current_obj.decode('utf-8')  # Convert bytes to a string
                }
                if parent is None:
                    serialized_obj = serialized_bytes
                else:
                    parent[current_obj[0]] = serialized_bytes

            elif isinstance(current_obj, complex):
                serialized_complex = {
                    "_object": "complex",
                    "real": current_obj.real,
                    "imag": current_obj.imag
                }
                if parent is None:
                    serialized_obj = serialized_complex
                else:
                    parent[current_obj[0]] = serialized_complex

            elif isinstance(current_obj, (int, float, str, bool)):
                serialized_primitive = {
                    "_object": type(current_obj).__name__,
                    "_payload": current_obj
                }
                if parent is None:
                    serialized_obj = serialized_primitive
                else:
                    parent[current_obj[0]] = serialized_primitive

            elif current_obj is None:
                serialized_none = {
                    "_object": "NoneType",
                    "_payload": None
                }
                if parent is None:
                    serialized_obj = serialized_none
                else:
                    parent[current_obj[0]] = serialized_none

            elif hasattr(current_obj, '__dict__'):
                class_name = current_obj.__class__.__name__
                class_module = current_obj.__class__.__module__

                # Register the class in the registry table
                self._register_class(class_name, class_module)

                serialized_class = {
                    "_object": "class",
                    "_name": class_name,
                    "_payload": {},
                    "module": class_module,
                }
                for k, v in current_obj.__dict__.items():
                    stack.append((serialized_class["_payload"], (k, v)))
                if parent is None:
                    serialized_obj = serialized_class
                else:
                    parent[current_obj[0]] = serialized_class

            else:
                serialized_unknown = {
                    "_object": "unknown",
                    "_payload": str(current_obj)
                }
                if parent is None:
                    serialized_obj = serialized_unknown
                else:
                    parent[current_obj[0]] = serialized_unknown

        return serialized_obj



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

    def _deserialize_object(self, data):
        stack = [(None, data)]
        deserialized_obj = {}

        while stack:
            parent, current_data = stack.pop()

            if isinstance(current_data, dict):
                obj_type = current_data.get("_object")

                if obj_type == "class":
                    class_name = current_data["_name"]
                    fields = current_data["_payload"]

                    # Look up the class module from the registry table
                    class_module = current_data.get("module")
                    if class_module:
                        module = __import__(class_module)
                        cls = getattr(module, class_name)
                        instance = cls.__new__(cls)
                        if parent is None:
                            deserialized_obj = instance
                        else:
                            parent[current_data[0]] = instance
                        stack.append((instance.__dict__, fields))

                    else:
                        raise ImportError(f"Class {class_name} not found in registry")

                elif obj_type in {"tuple", "set", "list"}:
                    container = {
                        "tuple": tuple(),
                        "set": set(),
                        "list": list()
                    }[obj_type]
                    if parent is None:
                        deserialized_obj = container
                    else:
                        parent[current_data[0]] = container
                    for item in reversed(current_data["_payload"]):
                        stack.append((container, item))

                elif obj_type == "bytes":
                    result = bytes(current_data["_payload"], 'utf-8')
                    if parent is None:
                        deserialized_obj = result
                    else:
                        parent[current_data[0]] = result

                elif obj_type == "complex":
                    result = complex(current_data["real"], current_data["imag"])
                    if parent is None:
                        deserialized_obj = result
                    else:
                        parent[current_data[0]] = result

                elif obj_type == "dict":
                    deserialized_dict = {}
                    if parent is None:
                        deserialized_obj = deserialized_dict
                    else:
                        parent[current_data[0]] = deserialized_dict
                    for k, v in reversed(current_data["_payload"].items()):
                        deserialized_key = json.loads(k)
                        stack.append((deserialized_dict, (deserialized_key, v)))

                elif obj_type in {"int", "float", "str", "bool", "NoneType"}:
                    result = {
                        "int": int,
                        "float": float,
                        "str": str,
                        "bool": bool,
                        "NoneType": lambda x: None
                    }[obj_type](current_data["_payload"])
                    if parent is None:
                        deserialized_obj = result
                    else:
                        parent[current_data[0]] = result

                else:
                    result = current_data["_payload"]
                    if parent is None:
                        deserialized_obj = result
                    else:
                        parent[current_data[0]] = result

        return deserialized_obj

    
    
    
    def _deserialize_object_recursive(self, data):
        if isinstance(data, dict):
            obj_type = data.get("_object")
            if obj_type == "class":
                class_name = data["_name"]
                fields = self._deserialize_object(data["_payload"])

                # Look up the class module from the registry table
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
            elif obj_type == "int":
                return int(data["_payload"])
            elif obj_type == "float":
                return float(data["_payload"])
            elif obj_type == "str":
                return str(data["_payload"])
            elif obj_type == "bool":
                return bool(data["_payload"])
            elif obj_type == "NoneType":
                return None
            else:
                return data["_payload"]
        else:
            return data



    def _get_class_module(self, class_name):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT class_module FROM class_registry WHERE class_name = ?", (class_name,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return None

    def _serialize_params(self, args, kwargs):
        serialized_args = [self._serialize_object(arg) for arg in args]
        serialized_kwargs = {key: self._serialize_object(value) for key, value in kwargs.items()}
        return json.dumps({"args": serialized_args, "kwargs": serialized_kwargs}, sort_keys=True)

    def _deserialize_params(self, params):
        data = json.loads(params)
        args = [self._deserialize_object(arg) for arg in data["args"]]
        kwargs = {key: self._deserialize_object(value) for key, value in data["kwargs"].items()}
        return args, kwargs

    def cached_function(self, func, threshold_time=0.05):
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

            # Cache the result only if execution time exceeds the threshold or if the threshold is 0
            if threshold_time == 0 or execution_time > threshold_time:
                self._set_cache(table_name, params, output, execution_time)
            return output
        return wrapper
    
    def cached_model(self, func, threshold_time=0.05):
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
    
    def cache_class_methods(self, cls, threshold_time=0.05):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value) and not attr_name.startswith('_'):
                wrapped_method = self.cached_function(attr_value, threshold_time)
                setattr(cls, attr_name, wrapped_method)
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
