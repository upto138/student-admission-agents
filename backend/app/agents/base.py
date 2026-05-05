"""
Base Agent abstraction for all agents
Using agent-framework pattern
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
# from typing import Any, Optional
 
from app.schemas.agent_state import AgentState
 
logger = logging.getLogger(__name__) # __name__: app.agents.base

class BaseAgent(ABC):
    """
    The Base class for all agents in the system
    Provide a interface so that Orchestrator can call it consistently.
    """
 
    name: str = "base_agent"
    description: str = "Base agent"
 
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._logger = logging.getLogger(f"agent.{self.name}")
 
    @abstractmethod
    async def run(self, state: AgentState) -> AgentState:
        """
        Run the agent with the current state and return the updated state
 
        Args:
            state: The state is shared between agents
 
        Returns:
            AgentState has been updated
        """
        ...
 
    def log(self, message: str, level: str = "info") -> None: # Todo: enumerate the levels of log - info, warning, error,...
        """Helper log can be enable/disable via verbose mode."""
        if self.verbose:
            getattr(self._logger, level)(f"[{self.name.upper()}] {message}")