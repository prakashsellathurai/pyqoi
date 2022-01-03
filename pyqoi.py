##### IMPORTS #######
from dataclasses import dataclass
from io import BytesIO
from typing import List, ByteString, Optional
import numpy as np
import os

### CONSTANTS ####
QOI_SRGB = 0
QOI_LINEAR = 1

QOI_ZEROARR = lambda a: [0 for _ in range(len(a))]
QOI_OP_INDEX = '0x00'
QOI_OP_DIFF = '0x40'
QOI_OP_LUMA = '0x80'
QOI_OP_RUN = '0xc0'
QOI_OP_RGB = '0xfe'
QOI_OP_RGBA = '0xfe'

QOI_MASK_2 = '0xc0'

QOI_COLOR_HASH = lambda C: C.rgba.r*3+C.rgba.g*5+C.rgba.b*7+C.rgba.a*11
QOI_MAGIC = ord('q') << 24 | ord('o') << 16 | ord('i') << 8 | ord('f')  
QOI_HEADER_SIZE = 14
QOI_PIXELS_MAX = 400000000

qoi_padding = [0 for _ in range(7)] + [1]


###  CLASSSES ####
@dataclass
class QoiHeader:
    width: np.uint32 # image width 
    height: np.uint32 # image height 
    channels: np.uint8 # 3 if RGB ,4 if RGBA
    colorspace: np.uint8 # 0 = sRGB with linear alpha, 1 = all channels linear

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
def qoiWrite32( bytes:List[str],p: int,v:np.uint):
    bytes[p] = (int('ff000000',16)& v) >> 24
    p+=1
    bytes[p] = (int('00ff0000',16)& v) >> 16
    p+=1
    bytes[p] = (int('0000ff00',16)& v) >> 8
    p+=1
    bytes[p] = (int('000000ff',16)& v)
    p+=1
    return bytes,p


def qoiRead32(bytes: List[str],p: int) -> np.uint:
    a = bytes[p]
    b = bytes[p+1]
    c = bytes[p+2]
    d = bytes[p+4]
    p += 4
    return a<<24 |b<<16 | c<<8 | d


##### IO #################



def encode(data:bytes,desc: QoiHeader, out_len: int) -> List[str]:
    """Encodes Numpy array into Qoi Format"""
    
    if (data is None or out_len is None or desc is None or
        desc.width == 0 or desc.height == 0 or 
        desc.channels < 3 or desc.channels > 4 or desc.colorspace > 1 or desc.height >= QOI_PIXELS_MAX / desc.width):
        return None

    max_size = desc.width * desc.height * (desc.channels + 1)+ QOI_HEADER_SIZE + len(qoi_padding)

    p = 0
    bytes = ['']*max_size

    bytes,p = qoiWrite32(bytes,p,QOI_MAGIC)
    bytes,p = qoiWrite32(bytes,p,desc.width)
    bytes,p = qoiWrite32(bytes,p,desc.height)

    bytes[p] = desc.channels
    p+=1
    bytes[p] = desc.colorspace
    p+=1

    pixels = bytes.copy()
    index: List[QoiRGBA] = [QoiRGBA()] * (64)
    run = 0
    px_prev: QoiRGBA =QoiRGBA(rgba=RGBA(r=0,g=0,b=0,a=255))
    px = px_prev

    px_len = desc.width * desc.height *desc.channels
    px_end = px_len - desc.channels
    channels = desc.channels

    for px_pos in range(0,px_len,channels):
        if channels == 4:
            px.rgba = pixels[px_pos]
        else:
            px.rgba.r = pixels[px_pos+0]
            px.rgba.g = pixels[px_pos + 1]
            px.rgba.b = pixels[px_pos+2]


    return pixels


def decode(bytes: ByteString, size: int):
    """Decodes Qoi data into Numpy Array"""
    pass


def read(filename: str, channels: Optional[int] = 0) -> np.ndarray:
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

    bytes: ByteString = f.read(size)

    pixels = decode(bytes, size)

    f.close()
    return pixels


def write(filename: str,data: bytes) -> None:
    """writes the Qoi Image to a file"""
    f = open(filename,'wb')
    encoded = encode(data)
    f.write(encode)
    f.close()


