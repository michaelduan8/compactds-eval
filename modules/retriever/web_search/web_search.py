'''
Web Search Abstract Class
'''
from abc import ABC, abstractmethod

class WebSearch(ABC):
    @abstractmethod
    def retrieve(self, queries_with_metadata):
        pass