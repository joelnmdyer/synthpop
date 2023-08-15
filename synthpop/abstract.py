from abc import ABC, abstractmethod
from dataclasses import dataclass

class AbstractModel(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def initialize(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def observe(self, *args, **kwargs):
        pass

    def __call__(self, generator):
        return self.run(generator)

class AbstractGenerator(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class AbstractMetaGenerator(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
