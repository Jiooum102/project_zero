from enum import StrEnum, auto


class GRImageType(StrEnum):
    numpy = auto()
    pil = auto()
    filepath = auto()


class GRImageMode(StrEnum):
    # 1 = auto()
    L = auto()
    P = auto()
    RGB = auto()
    RGBA = auto()
    CMYK = auto()
    YCbCr = auto()
    LAB = auto()
    HSV = auto()
    I = auto()
    F = auto()
