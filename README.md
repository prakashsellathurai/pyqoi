# PyQOI

A Python implementation of the QOI (Quite OK Image) format.

## About QOI

QOI (Quite OK Image format) is a fast, lossless image compression format. Its compression ratios are similar to PNG, but it offers significantly faster encoding and decoding. For more information about the QOI format, visit the [official QOI website](https://qoiformat.org/).

## Installation

```bash
pip install pyqoi
```

## Usage

### Reading a QOI image

```python
from pyqoi import QoiHeader, read
import numpy as np
from PIL import Image

# Create a header object
header = QoiHeader(width=0, height=0, channels=0, colorspace=0)

# Read pixels from a QOI file
pixels = read("image.qoi", header)

if pixels is not None:
    # Convert to numpy array for further processing or displaying with PIL
    pixels_array = np.frombuffer(pixels, dtype=np.uint8)
    pixels_array = pixels_array.reshape((header.height, header.width, header.channels))
    
    # Create PIL image
    if header.channels == 3:
        img = Image.fromarray(pixels_array, mode="RGB")
    else:
        img = Image.fromarray(pixels_array, mode="RGBA")
    
    img.show()
```

### Writing a QOI image

```python
from pyqoi import QoiHeader, write
from PIL import Image
import numpy as np

# Load an image with PIL
pil_img = Image.open("image.png")


# Convert to RGB or RGBA
if pil_img.mode != "RGB" and pil_img.mode != "RGBA":
    pil_img = pil_img.convert("RGBA")

# Get image data as numpy array
img_array = np.array(pil_img)
channels = 4 if pil_img.mode == "RGBA" else 3

# Create QOI header
header = QoiHeader(
    width=pil_img.width,
    height=pil_img.height,
    channels=channels,
    colorspace=0  # 0 for sRGB with linear alpha
)

# Get bytes from numpy array
pixels = img_array.tobytes()

# Write to QOI file
write("output.qoi", pixels, header, len(pixels))
```

## API Reference

### Classes

#### `QoiHeader`

```python
@dataclass
class QoiHeader:
    width: np.uint32  # image width
    height: np.uint32  # image height
    channels: np.uint8  # 3 if RGB, 4 if RGBA
    colorspace: np.uint8  # 0 = sRGB with linear alpha, 1 = all channels linear
```

### Functions

#### `read(filename, desc, channels=0)`

Reads a QOI image file and decodes it to raw pixel data.

- `filename`: Path to the QOI file
- `desc`: A `QoiHeader` object that will be populated with image information
- `channels`: Optional. Number of channels to use (0 to use the file's native channels)
- Returns: A bytes object containing the raw pixel data

#### `write(filename, data, desc, out_len)`

Encodes raw pixel data and writes it to a QOI file.

- `filename`: Output path for the QOI file
- `data`: Raw pixel data as bytes
- `desc`: A `QoiHeader` object with image information
- `out_len`: Length of the pixel data in bytes

#### `encode(data, desc, out_len)`

Encodes raw pixel data to QOI format.

- `data`: Raw pixel data as bytes
- `desc`: A `QoiHeader` object with image information
- `out_len`: Length of the pixel data in bytes
- Returns: A tuple of (encoded_data, encoded_length)

#### `decode(data, size, desc, channels=0)`

Decodes QOI format data to raw pixels.

- `data`: QOI encoded data as bytes
- `size`: Size of the encoded data
- `desc`: A `QoiHeader` object that will be populated with image information
- `channels`: Optional. Number of channels to use (0 to use the file's native channels)
- Returns: A bytes object containing the raw pixel data

## License

MIT