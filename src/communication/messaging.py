"""Explicit, non-free message passing between robots (RACA-Collective.txt Section 14).

Every delegation request/response is a Message with an explicit latency cost and counts
against the fleet's message budget. There is no free channel: sending a delegation
request costs one message and `msg_latency` seconds, and a response costs another
message and `msg_latency` seconds, so a full delegate round-trip costs 2 messages and
`2 * msg_latency` seconds of communication latency, on top of whatever latency the peer's
own reasoning backend takes.
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class Message:
    sender: int
    receiver: int
    kind: str  # "delegation_request" | "delegation_response" | "reject"
    payload_size: int  # candidate count, a stand-in for bytes-on-the-wire
    latency: float


class MessageBus:
    """Tracks communication cost. `msg_latency` is the one-way transmission delay."""

    def __init__(self, msg_latency: float = 0.05, msg_latency_std: float = 0.01):
        self.msg_latency = msg_latency
        self.msg_latency_std = msg_latency_std
        self.messages: list[Message] = []
        self.total_messages = 0
        self.total_latency = 0.0
        self.rejected = 0
        self.duplicate_requests = 0

    def send(self, sender: int, receiver: int, kind: str, payload_size: int, rng) -> Message:
        lat = max(0.001, rng.normal(self.msg_latency, self.msg_latency_std))
        msg = Message(sender=sender, receiver=receiver, kind=kind, payload_size=payload_size, latency=lat)
        self.messages.append(msg)
        self.total_messages += 1
        self.total_latency += lat
        if kind == "reject":
            self.rejected += 1
        return msg

    def record_duplicate(self):
        self.duplicate_requests += 1

    def summary(self) -> dict:
        return {
            "total_messages": self.total_messages,
            "total_message_latency": self.total_latency,
            "rejected_requests": self.rejected,
            "duplicate_requests": self.duplicate_requests,
        }
