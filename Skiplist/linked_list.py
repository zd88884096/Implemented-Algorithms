from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class LinkedListNode(Generic[T]):
    data: T
    left: LinkedListNode[T] | None = None
    right: LinkedListNode[T] | None = None
    above: LinkedListNode[T] | None = None
    below: LinkedListNode[T] | None = None

    def __str__(self) -> str:
        return f"LinkedListNode: {self.data}"


@dataclass
class LinkedList(Generic[T]):
    head: LinkedListNode[T] | None = None
    tail: LinkedListNode[T] | None = None
