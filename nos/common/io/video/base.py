"""Simple video reader."""
from abc import ABC, abstractmethod
from typing import Iterator, Optional, TypeVar


T = TypeVar("T")


class BaseVideoReader(ABC):
    def __repr__(self) -> str:
        return f"{self.__class__.__name__} [fn={self.filename}]"

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        raise NotImplementedError()

    @abstractmethod
    def __next__(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, idx: int) -> T:
        raise NotImplementedError()

    @abstractmethod
    def open(self) -> T:
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def pos(self) -> Optional[int]:
        raise NotImplementedError()

    @abstractmethod
    def seek(self, idx: int) -> None:
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        self.seek(0)
