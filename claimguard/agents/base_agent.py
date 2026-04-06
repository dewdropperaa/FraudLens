from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAgent(ABC):
    def __init__(self, name: str, role: str, goal: str):
        self.name = name
        self.role = role
        self.goal = goal

    @abstractmethod
    def analyze(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
