
from typing import Generic, MutableMapping, TypeVar

T = TypeVar("T")

class Config(Generic[T]):
    def __init__(self, parent=None):
        self.data = {}
        self.parent = None

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value
        self.persist_data()

    def __delitem__(self, key):
        del self._data[key]
        self.persist_data()
    
    def __getattr__(self, key):
        return self._data[key]
    
    def __setattr__(self, key, value):
        self.data[key] = value
        self.persist_data()
    
    def __delattr__(self, key):
        del self._data[key]
        self.persist_data()
    
    def persist_data(self):
        if self.parent is None:
            self.write_data()
        else:
            self.parent.write_data()
    
    def write_data(self):
        pass


