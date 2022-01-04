##### IMPORTS #######
from dataclasses import dataclass
from io import BytesIO
from typing import List, ByteString, Optional, Tuple
import numpy as np
import os, sys

### CONSTANTS ####
QOI_SRGB = 0
QOI_LINEAR = 1

QOI_ZEROARR = lambda a: [0 for _ in range(len(a))]
QOI_OP_INDEX = 0x00
QOI_OP_DIFF = 0x40
QOI_OP_LUMA = 0x80
QOI_OP_RUN = 0xC0
QOI_OP_RGB = 0xFE
QOI_OP_RGBA = 0xFE

QOI_MASK_2 = 0xC0

QOI_COLOR_HASH = lambda C: C.rgba.r * 3 + C.rgba.g * 5 + C.rgba.b * 7 + C.rgba.a * 11
QOI_MAGIC = ord("q") << 24 | ord("o") << 16 | ord("i") << 8 | ord("f")
QOI_HEADER_SIZE = 14
QOI_PIXELS_MAX = 400000000

qoi_padding = [0 for _ in range(7)] + [1]


###  CLASSSES ####
@dataclass
class QoiHeader:
    width: np.uint32  # image width
    height: np.uint32  # image height
    channels: np.uint8  # 3 if RGB ,4 if RGBA
    colorspace: np.uint8  # 0 = sRGB with linear alpha, 1 = all channels linear


@dataclass
class RGBA:
    r: int
    g: int
    b: int
    a: int


@dataclass
class QoiRGBA:
    rgba: RGBA = None
    v: np.uint = None


##### Util Functions ####
def qoiWrite32(bytes: List[str], p: int, v: np.uint):
    bytes[p] = (0xFF000000 & v) >> 24
    p += 1
    bytes[p] = (0x00FF0000 & v) >> 16
    p += 1
    bytes[p] = (0x0000FF00 & v) >> 8
    p += 1
    bytes[p] = 0x000000FF & v
    p += 1
    return bytes, p


def qoiRead32(bytes: List[str], p: int) -> Tuple[np.uint, int]:
    a = bytes[p]
    p += 1
    b = bytes[p]
    p += 1
    c = bytes[p]
    p += 1
    d = bytes[p]
    p += 1
    return (a << 24 | b << 16 | c << 8 | d, p)


##### IO #################


def encode(data: List[str], desc: QoiHeader, out_len: int) -> Tuple[List[str], int]:
    """Encodes Raw RGB Pixels into Qoi Format

    Args:
        data (bytes): Raw RGB/RGBA data
        desc (QoiHeader): QoiHeader data
        out_len (int): Raw RGB/RGBA  data length

    Returns:
        List[int]: encoded data
    """

    if (
        data is None
        or out_len is None
        or desc is None
        or desc.width == 0
        or desc.height == 0
        or desc.channels < 3
        or desc.channels > 4
        or desc.colorspace > 1
        or desc.height >= QOI_PIXELS_MAX / desc.width
    ):
        return None

    max_size = (
        desc.width * desc.height * (desc.channels + 1)
        + QOI_HEADER_SIZE
        + sys.getsizeof(qoi_padding)
    )

    p = 0
    encoded: List[str] = [0] * max_size

    encoded, p = qoiWrite32(encoded, p, QOI_MAGIC)
    encoded, p = qoiWrite32(encoded, p, desc.width)
    encoded, p = qoiWrite32(encoded, p, desc.height)

    encoded[p] = desc.channels
    p += 1
    encoded[p] = desc.colorspace
    p += 1

    pixels = data
    index: List[QoiRGBA] = [QoiRGBA()] * (64)
    run = 0
    px_prev: QoiRGBA = QoiRGBA(rgba=RGBA(r=0, g=0, b=0, a=255))
    px = px_prev

    px_len = desc.width * desc.height * desc.channels
    px_end = px_len - desc.channels
    channels = desc.channels

    for px_pos in range(0, px_len, channels):
        if channels == 4:
            # TODO work on better way to do this
            # px = QoiRGBA(rgba=pixels[px_pos:channels])
            px.rgba.r = pixels[px_pos + 0]
            px.rgba.g = pixels[px_pos + 1]
            px.rgba.b = pixels[px_pos + 2]
            px.rgba.a = pixels[px_pos + 3]
        else:
            px.rgba.r = pixels[px_pos + 0]
            px.rgba.g = pixels[px_pos + 1]
            px.rgba.b = pixels[px_pos + 2]

        if px.v == px_prev.v:
            run += 1
            if run == 62 or px_pos == px_end:
                encoded[p] = QOI_OP_RUN | (run - 1)
                p += 1
                run = 0
        else:
            if run > 0:
                encoded[p] = QOI_OP_RUN | (run - 1)
                p += 1
            index_pos = QOI_COLOR_HASH(px) % 64

            if index[index_pos].v == px.v:
                encoded[p] = QOI_OP_INDEX | index_pos
            else:
                index[index_pos] = px

                if px.rgba.a == px_prev.rgba.a:
                    vr = px.rgba.r - px_prev.rgba.r
                    vg = px.rgba.g - px_prev.rgba.g
                    vb = px.rgba.b - px_prev.rgba.b

                    vg_r = vr - vg
                    vg_b = vb - vg

                    if -3 < vr < 2 and -3 < vg < 2 and -3 < vb < 2:
                        encoded[p] = (
                            QOI_OP_DIFF | (vr + 2) << 4 | (vg + 2) << 2 | (vb + 2)
                        )
                        p += 1
                    elif -9 < vg_r < 8 and -33 < vg < 32 and -9 < vg_b < 8:
                        encoded[p] = QOI_OP_LUMA | (vg + 32)
                        p += 1
                        encoded[p] = (vg_r + 8) << 4 | (vg_b + 8)
                        p += 1
                    else:
                        encoded[p] = QOI_OP_RGB
                        p += 1
                        encoded[p] = px.rgba.r
                        p += 1
                        encoded[p] = px.rgba.g
                        p += 1
                        encoded[p] = px.rgba.b
                        p += 1
                else:
                    encoded[p] = QOI_OP_RGBA
                    p += 1
                    encoded[p] = px.rgba.r
                    p += 1
                    encoded[p] = px.rgba.g
                    p += 1
                    encoded[p] = px.rgba.b
                    p += 1
                    encoded[p] = px.rgba.a
                    p += 1

        px_prev = px

    for i in range(len(qoi_padding)):
        encoded[p] = qoi_padding[i]
        p += 1
    out_len = p
    return encoded, out_len


def decode(data: List[str], size: int, desc: QoiHeader, channels: int = 0):
    """Decodes Encoded Qoi Image into Raw pixels"""
    p: int = 0
    run: int = 0

    if (
        data is None
        or desc is None
        or (channels != 0 and channels != 3 and channels != 4)
        or size < QOI_HEADER_SIZE + sys.getsizeof(qoi_padding)
    ):
        return None

    decoded: List[str] = data.__str__()

    header_magic, p = qoiRead32(data, p)
    desc.width, p = qoiRead32(data, p)
    desc.height, p = qoiRead32(data, p)
    desc.channels = data[p]
    p += 1
    desc.colorspace = data[p]
    p += 1

    if (
        desc.width == 0
        or desc.height == 0
        or desc.channels < 3
        or desc.channels > 4
        or desc.colorspace > 1
        or header_magic != QOI_MAGIC
        or desc.height >= QOI_PIXELS_MAX / desc.width
    ):
        return None

    if channels == 0:
        channels = desc.channels

    px_len: int = desc.width * desc.height * channels
    pixels: List[str] = [0] * px_len

    if not pixels:
        return None

    index: List[QoiRGBA] = [QoiRGBA()] * (64)
    px = QoiRGBA(rgba=RGBA(r=0, g=0, b=0, a=255))

    chunks_len = size - int(sys.getsizeof(qoi_padding))

    for px_pos in range(0, px_len, channels):
        if run > 0:
            run -= 1
        elif p < chunks_len:
            b1: int = data[p]
            p += 1

            if b1 == QOI_OP_RGB:
                px.rgba.r = bytes[p]
                p += 1
                px.rgba.g = bytes[p]
                p += 1
                px.rgba.b = bytes[p]
                p += 1
            elif b1 == QOI_OP_RGBA:
                px.rgba.r = bytes[p]
                p += 1
                px.rgba.g = bytes[p]
                p += 1
                px.rgba.b = bytes[p]
                p += 1
                px.rgba.a = bytes[p]
                p += 1
            elif (b1 & QOI_MASK_2) == QOI_OP_INDEX:
                px = index[b1]
            elif (b1 & QOI_MASK_2) == QOI_OP_DIFF:
                px.rgba.r += ((b1 >> 4) & 0x03) - 2
                px.rgba.g += ((b1 >> 2) & 0x03) - 2
                px.rgba.b += (b1 & 0x03) - 2
            elif (b1 & QOI_MASK_2) == QOI_OP_LUMA:
                b2 = bytes[p]
                p += 1
                vg = (b1 & 0x3F) - 32
                px.rgba.r += vg - 8 + (b2 >> 4) & 0x0F
                px.rgba.g += vg
                px.rgba.b += vg - 8 + (b2 & 0x0F)
            elif (b1 & QOI_MASK_2) == QOI_OP_RUN:
                run = b1 & 0x3F
            index[QOI_COLOR_HASH(px) % 64] = px

        if channels == 4:
            # TODO work on better way to do this
            pixels[px_pos + 0] = px.rgba.r
            pixels[px_pos + 1] = px.rgba.g
            pixels[px_pos + 2] = px.rgba.b
            pixels[px_pos + 3] = px.rgba.a

        else:
            pixels[px_pos + 0] = px.rgba.r
            pixels[px_pos + 1] = px.rgba.g
            pixels[px_pos + 2] = px.rgba.b

    return pixels


def read(filename: str, desc: QoiHeader, channels: Optional[int] = 0) -> np.ndarray:
    """Reads a Qoi Image from a file

    Args:
        filename (str):
        channels (Optional[int]): [color channel]

    Returns:
        np.ndarray: Numpy array representing pixel data
    """

    # check is the file exists
    if not os.path.isfile(filename):
        print("File not Found Error")
        return

    # create file object f
    f = open(filename, "rb")

    # seek the end of the file
    f.seek(0, 2)

    # get the file size
    size: int = f.tell()

    # go to the beginnning of the file again
    f.seek(0)

    bytes = f.read(size)

    pixels = decode(bytes, size, desc)

    f.close()
    return pixels


def write(filename: str, data: bytes, desc: QoiHeader) -> None:
    """writes the Qoi Image to a file"""
    f = open(filename, "wb")
    out_len = len(data)
    encoded, out_len = encode(data, desc, out_len)
    f.write(encoded)
    f.close()
