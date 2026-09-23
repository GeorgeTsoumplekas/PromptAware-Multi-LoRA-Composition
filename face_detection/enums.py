from enum import Enum


class NetworkSize(Enum):
    LARGE = 4

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        return member

    def __int__(self):
        return self.value


class LandmarksType(Enum):
    """Enum class defining the type of landmarks to detect."""

    _2D = 1
    _2halfD = 2
    _3D = 3
