import traceback
from typing import Generic, MutableMapping, TypeVar
import yaml
from config_loader import MelodeusConfig
import dataclasses
import json

T = TypeVar("T")

class PersistentMelodeusConfig(Generic[T]):
    def __init__(self, data=None, parent=None, path=None):
        self.data = {} if data is None else data
        self.parent = parent
        self.path = path

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]
    
    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError as e:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {key}") from None
    
    def __setattr__(self, key, value):
        # avoid infinite recursion on member fields
        if key in ("data", "parent", "path"):
            object.__setattr__(self, key, value)
            return
        self.data[key] = value
    
    def __delattr__(self, key):
        try:
            del self.data[key]
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {key}") from None
    
    def to_json(self):
        return json.dumps(self.to_dict())

    def to_dict(self):
        res = {}
        for k,v in self.data.items():
            if type(v) is dict:
                res[k] = {ki: (vi.to_dict() if hasattr(vi, "to_dict") else vi) for (ki,vi) in v.items()}
            elif type(v) is list:
                res[k] = [(vi.to_dict() if hasattr(vi, "to_dict") else vi) for vi in v]
            else:
                res[k] = v.to_dict() if hasattr(v, "to_dict") else v
        return res
    
    @classmethod
    def from_dict(cls, d, parent=None, path=None):
        if d is None:
            return cls(data={}, parent=parent, path=path)
        else:
            res = cls(parent=parent, path=path)
            converted_dict = {}
            for k,v in d.items():
                if type(v) is list:
                    converted_dict[k] = [(cls.from_dict(vi, parent=res) if type(vi) is dict else vi) for vi in v]
                else:
                    converted_dict[k] = cls.from_dict(v, parent=res) if type(v) is dict else v
            res.data = converted_dict
            return res

    def persist_data(self):
        print(f"Persisting to {self.path} parent {self.parent}")
        # write uppermost parent so we store everything
        if self.parent is None:
            self.write_config()
        else:
            self.parent.persist_data()
    
    def write_config(self):
        data = self.to_dict()
        if self.path is not None:
            with open(str(self.path), "w") as f:
                try:
                    data_yaml = yaml.safe_dump(data)
                    f.write(data_yaml)
                except yaml.YAMLError as e:
                    print("Error writing config to yaml")
                    print(traceback.print_exc())
    
    @classmethod
    def load_config(cls, path):
        # do melodeus config so all fields are populated
        melodeus_config = None
        try:
            with open(path, "r") as f:
                try:
                    json_data = yaml.safe_load(f.read())
                    if json_data is not None:
                        melodeus_config = MelodeusConfig(**json_data)
                except yaml.YAMLError as e:
                    print("Error writing loading config, reading blank file")
                    print(traceback.print_exc())
        except FileNotFoundError:
            pass # fill in defaults below
        no_config = melodeus_config is None
        melodeus_config = MelodeusConfig() if melodeus_config is None else melodeus_config
        config_dict = dataclasses.asdict(melodeus_config)
        res = cls.from_dict(config_dict)
        res.path = path
        if no_config:
            res.persist_data()
        return res