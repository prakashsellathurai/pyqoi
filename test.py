from typing import List
import unittest
import pyqoi as pqoi
import numpy as np
from PIL import Image


class TestEncodingAndDecoding(unittest.TestCase):
    def test_encode(self):
        f = Image.open("./qoi_test_images/dice.png")
        data: List[str] = [item for t in list(f.getdata()) for item in t]
        out_len = len(data)

        desc = pqoi.QoiHeader(
            width=f.width,
            height=f.height,
            channels=len(f.getbands()),
            colorspace=f.getexif().get("ColorSpace", 0),
        )
        

        encoded,out_len = pqoi.encode(data, desc, out_len)
        decoded = pqoi.decode(encoded,len(encoded),desc)
        
        
        

        f.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
