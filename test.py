import unittest
import pyqoi as pqoi
import numpy as np
from PIL import Image


class TestEncodingAndDecoding(unittest.TestCase):
    def test_encode(self):
        assert 1 == 1
        f = Image.open("./qoi_test_images/dice.png")
        data = f.tobytes()
        width = f.width
        height = f.height
        channels = len(f.getbands())
        colorspace = f.getexif().get('ColorSpace',0)
        out_len = len(data)
        
        pqoi.encode(
            data,
            pqoi.QoiHeader(
                width=width, height=height, channels=channels, colorspace=colorspace
            ),
            out_len
        )
        f.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
