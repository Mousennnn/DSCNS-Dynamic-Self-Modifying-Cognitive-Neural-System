"""Dynamic Self-Modifying Cognitive Network System (DSCNS).

A faithful implementation of the DSCNS design report:
    experience -> multi-network observation -> independent evaluation ->
    cross-network verification -> selective internalization ->
    structural evolution -> continuous learning.

Phase 0 + Phase 1 prototype scope: a 100M-500M base language model with
5-6 cognitive networks, continual-learning experiments (Control / Exp1 / Exp2).
"""

__version__ = "0.1.0"
__all__ = [
    "DSCNSSystem",
    "BaseLanguageModel",
    "CognitiveNetwork",
    "VerificationNetwork",
    "InternalizationController",
    "MetaCognitiveController",
    "MemorySystem",
    "NetworkCommunicationBus",
]
