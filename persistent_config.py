import traceback
from typing import Generic, MutableMapping, TypeVar
import yaml

T = TypeVar("T")

class PersistentConfig(Generic[T]):
    def __init__(self, data=None, parent=None, path=None):
        self.data = {} if data is None else data
        self.parent = parent
        self.path = path

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
        self.persist_data()

    def __delitem__(self, key):
        del self.data[key]
        self.persist_data()
    
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
        self.persist_data()
    
    def __delattr__(self, key):
        try:
            del self.data[key]
            self.persist_data()
        except KeyError:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {key}") from None
    
    def to_dict(self):
        res = {}
        for k,v in self.data.items():
            res[k] = v.to_dict() if hasattr(v, "to_dict") else v
        return res
    
    @classmethod
    def from_dict(cls, d, parent=None, path=None):
        if d is None:
            return cls(data={}, parent=parent)
        else:
            res = cls(parent=parent, path=path)
            converted_dict = {}
            for k,v in d.items():
                converted_dict[k] = cls.from_dict(v, parent=res) if type(v) is dict else v
            res.data = converted_dict
            return res

    def persist_data(self):
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
        json_data = {}
        with open(path, "r") as f:
            try:
                json_data = yaml.safe_load(f.read())
            except yaml.YAMLError as e:
                print("Error writing loading config, reading blank file")
                print(traceback.print_exc())
        res = PersistentConfig.from_dict(json_data)
        res.path = path
        return res
