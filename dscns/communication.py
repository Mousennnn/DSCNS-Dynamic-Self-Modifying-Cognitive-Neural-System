"""Network communication protocol (report section 8.1).

Implements NetworkMessage, MessageType and the async NetworkCommunicationBus
with a fully traceable message log and a co-activation matrix used for
dynamic-connection decisions.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MessageType(Enum):
    QUERY = "query"  # active query of other networks' knowledge
    RESPONSE = "response"  # response to a query
    BROADCAST = "broadcast"  # broadcast candidate knowledge to all networks
    CONFLICT = "conflict"  # report knowledge conflict
    CONFIDENCE = "confidence"  # share confidence evaluation
    UPDATE_NOTIFY = "update_notify"  # notify that own parameters were updated
    CORRECTION = "correction"  # correct another network's evaluation
    MERGE_REQUEST = "merge_request"  # request a structural merge
    SPLIT_NOTIFY = "split_notify"  # notify that self has split
    META_REPORT = "meta_report"  # report state to the meta-cognitive layer


class NetworkMessage:
    """A message exchanged between cognitive networks."""

    __slots__ = ("sender", "receiver", "msg_type", "content", "timestamp")

    def __init__(
        self,
        sender: str,
        receiver: str,
        msg_type: MessageType,
        content: Any,
    ):
        self.sender = sender
        self.receiver = receiver  # network id or "broadcast"
        self.msg_type = msg_type
        self.content = content
        self.timestamp = time.time()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"NetworkMessage({self.sender}->{self.receiver}, "
            f"{self.msg_type.value}, ts={self.timestamp:.2f})"
        )


class NetworkCommunicationBus:
    """Asynchronous message bus connecting cognitive networks.

    The bus is fully traceable: every message is appended to ``message_log``,
    which also powers the co-activation matrix used by the structure-evolution
    layer (report section 4.4).
    """

    def __init__(self, networks: Optional[Dict[str, Any]] = None):
        self.networks: Dict[str, Any] = networks or {}
        # NOTE: plain deque instead of asyncio.Queue -- on Python 3.8 an
        # asyncio.Queue binds to a (possibly closed) event loop and breaks
        # after repeated asyncio.run() calls ("no current event loop").
        self.message_queue: deque = deque()
        self.message_log: List[NetworkMessage] = []
        self._subscribers: Dict[str, List[Any]] = defaultdict(list)

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #
    def register(self, network: Any) -> None:
        self.networks[network.id] = network

    def subscribe(self, network_id: str, network: Any) -> None:
        self._subscribers[network_id].append(network)

    # ------------------------------------------------------------------ #
    # async core (faithful to the design report)
    # ------------------------------------------------------------------ #
    async def send(self, message: NetworkMessage) -> None:
        """Send a message (broadcast or unicast)."""
        self.message_log.append(message)
        self.message_queue.append(message)

        if message.receiver == "broadcast":
            for net_id, network in self.networks.items():
                if net_id != message.sender:
                    await network.receive(message)
        else:
            receiver = self.networks.get(message.receiver)
            if receiver is not None:
                await receiver.receive(message)
            else:
                # receiver unknown -> fall back to subscribers
                for sub in self._subscribers.get(message.receiver, []):
                    await sub.receive(message)

    # ------------------------------------------------------------------ #
    # synchronous convenience used by the experiment harness
    # ------------------------------------------------------------------ #
    def send_sync(self, message: NetworkMessage) -> None:
        """Synchronous wrapper around :meth:`send`."""
        asyncio.run(self.send(message))

    # ------------------------------------------------------------------ #
    # statistics for structure evolution
    # ------------------------------------------------------------------ #
    def get_co_activation_matrix(self, window: int = 1000) -> Dict[Tuple[str, str], int]:
        """Count co-activation of network pairs from recent query traffic."""
        co_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
        recent_logs = self.message_log[-window:]
        for msg in recent_logs:
            if msg.msg_type in (MessageType.QUERY, MessageType.RESPONSE):
                co_matrix[(msg.sender, msg.receiver)] += 1
        return dict(co_matrix)

    def get_message_stats(self) -> Dict[str, int]:
        stats: Dict[str, int] = defaultdict(int)
        for msg in self.message_log:
            stats[msg.msg_type.value] += 1
        return dict(stats)
