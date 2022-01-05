from typing import List
import unittest
import pyqoi as pqoi
import numpy as np
from PIL import Image


class BasicTest(unittest.TestCase):
    # def test_basic_encode_and_decode(self):
    #     f = Image.open("./qoi_test_images/dice.png")
    #     data: List[str] = [item for t in list(f.getdata()) for item in t]
    #     out_len = len(data)

    #     desc = pqoi.QoiHeader(
    #         width=f.width,
    #         height=f.height,
    #         channels=len(f.getbands()),
    #         colorspace=f.getexif().get("ColorSpace", 0),
    #     )
        

    #     encoded,out_len = pqoi.encode(data, desc, out_len)
    #     decoded = pqoi.decode(encoded,len(encoded),desc)
    
    #     f.close()

    def test_file_conversion(self):
        f = Image.open("./qoi_test_images/dice.png")
        data: bytes =  f.tobytes()
        out_len = len(data)

        desc = pqoi.QoiHeader(
            width=f.width,
            height=f.height,
            channels=len(f.getbands()),
            colorspace=f.getexif().get("ColorSpace", 0),
        )
        pqoi.write('./test.qoi',data,desc,out_len)
        f.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
