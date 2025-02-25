import unittest
import os
import tempfile
import numpy as np
from io import BytesIO
from pyqoi import (
    QoiHeader, RGBA, QoiRGBA, 
    encode, decode, read, write,
    QOI_SRGB, QOI_LINEAR
)

class TestPyQOI(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and temporary directory for files"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple 2x2 RGB image
        self.rgb_data = bytearray([
            255, 0, 0,    # Red pixel
            0, 255, 0,    # Green pixel
            0, 0, 255,    # Blue pixel
            255, 255, 0   # Yellow pixel
        ])
        self.rgb_header = QoiHeader(width=2, height=2, channels=3, colorspace=QOI_SRGB)
        
        # Create a simple 2x2 RGBA image
        self.rgba_data = bytearray([
            255, 0, 0, 255,      # Red pixel (opaque)
            0, 255, 0, 255,      # Green pixel (opaque)
            0, 0, 255, 255,      # Blue pixel (opaque)
            255, 255, 0, 128     # Yellow pixel (semi-transparent)
        ])
        self.rgba_header = QoiHeader(width=2, height=2, channels=4, colorspace=QOI_SRGB)
        
        # Solid color test (all pixels same color)
        self.solid_rgb_data = bytearray([128, 128, 128] * 16)  # 4x4 gray image
        self.solid_rgb_header = QoiHeader(width=4, height=4, channels=3, colorspace=QOI_SRGB)
        
        # Gradient test
        self.gradient_data = bytearray()
        for i in range(256):
            self.gradient_data.extend([i, i, i])  # Grayscale gradient
        self.gradient_header = QoiHeader(width=16, height=16, channels=3, colorspace=QOI_SRGB)
        
        # Create test file paths
        self.rgb_file = os.path.join(self.temp_dir, "test_rgb.qoi")
        self.rgba_file = os.path.join(self.temp_dir, "test_rgba.qoi")
        self.solid_file = os.path.join(self.temp_dir, "test_solid.qoi")
        self.gradient_file = os.path.join(self.temp_dir, "test_gradient.qoi")
    
    def tearDown(self):
        """Clean up temporary files"""
        for file in [self.rgb_file, self.rgba_file, self.solid_file, self.gradient_file]:
            if os.path.exists(file):
                os.remove(file)
        os.rmdir(self.temp_dir)
    
    def test_rgb_encode_decode(self):
        """Test encoding and decoding of RGB data"""
        # Encode the RGB data
        encoded_data, encoded_len = encode(self.rgb_data, self.rgb_header, len(self.rgb_data))
        
        # Ensure encoding produced data
        self.assertIsNotNone(encoded_data)
        self.assertGreater(encoded_len, 0)
        
        # Create a new header for decoding
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # Decode the data
        decoded_data = decode(encoded_data, encoded_len, decode_header)
        
        # Verify the decoded header matches the original
        self.assertEqual(decode_header.width, self.rgb_header.width)
        self.assertEqual(decode_header.height, self.rgb_header.height)
        self.assertEqual(decode_header.channels, self.rgb_header.channels)
        self.assertEqual(decode_header.colorspace, self.rgb_header.colorspace)
        
        # Verify the decoded data matches the original
        self.assertEqual(len(decoded_data), len(self.rgb_data))
        for i in range(len(self.rgb_data)):
            self.assertEqual(decoded_data[i], self.rgb_data[i])
    
    def test_rgba_encode_decode(self):
        """Test encoding and decoding of RGBA data"""
        # Encode the RGBA data
        encoded_data, encoded_len = encode(self.rgba_data, self.rgba_header, len(self.rgba_data))
        
        # Ensure encoding produced data
        self.assertIsNotNone(encoded_data)
        self.assertGreater(encoded_len, 0)
        
        # Create a new header for decoding
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # Decode the data
        decoded_data = decode(encoded_data, encoded_len, decode_header)
        
        # Verify the decoded header matches the original
        self.assertEqual(decode_header.width, self.rgba_header.width)
        self.assertEqual(decode_header.height, self.rgba_header.height)
        self.assertEqual(decode_header.channels, self.rgba_header.channels)
        self.assertEqual(decode_header.colorspace, self.rgba_header.colorspace)
        
        # Verify the decoded data matches the original
        self.assertEqual(len(decoded_data), len(self.rgba_data))
        for i in range(len(self.rgba_data)):
            self.assertEqual(decoded_data[i], self.rgba_data[i])
    
    def test_solid_color_compression(self):
        """Test encoding and decoding of solid color images (tests run-length encoding)"""
        # Encode the solid color data
        encoded_data, encoded_len = encode(self.solid_rgb_data, self.solid_rgb_header, len(self.solid_rgb_data))
        
        # Solid color should compress very well
        self.assertLess(encoded_len, len(self.solid_rgb_data))
        
        # Create a new header for decoding
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # Decode the data
        decoded_data = decode(encoded_data, encoded_len, decode_header)
        
        # Verify the decoded data matches the original
        self.assertEqual(len(decoded_data), len(self.solid_rgb_data))
        for i in range(len(self.solid_rgb_data)):
            self.assertEqual(decoded_data[i], self.solid_rgb_data[i])
    
    def test_gradient_encoding(self):
        """Test encoding and decoding of gradient data (tests diff encoding)"""
        # Encode the gradient data
        encoded_data, encoded_len = encode(self.gradient_data, self.gradient_header, len(self.gradient_data))
        
        # Gradient should compress somewhat due to diff encoding
        self.assertLess(encoded_len, len(self.gradient_data))
        
        # Create a new header for decoding
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # Decode the data
        decoded_data = decode(encoded_data, encoded_len, decode_header)
        
        # Verify the decoded data matches the original
        self.assertEqual(len(decoded_data), len(self.gradient_data))
        for i in range(len(self.gradient_data)):
            self.assertEqual(decoded_data[i], self.gradient_data[i])
    
    def test_file_io_rgb(self):
        """Test writing and reading RGB QOI files"""
        # Write the RGB data to a file
        write(self.rgb_file, self.rgb_data, self.rgb_header, len(self.rgb_data))
        
        # Verify the file exists
        self.assertTrue(os.path.exists(self.rgb_file))
        
        # Create a new header for reading
        read_header = QoiHeader(0, 0, 0, 0)
        
        # Read the file
        read_data = read(self.rgb_file, read_header)
        
        # Verify the read header matches the original
        self.assertEqual(read_header.width, self.rgb_header.width)
        self.assertEqual(read_header.height, self.rgb_header.height)
        self.assertEqual(read_header.channels, self.rgb_header.channels)
        self.assertEqual(read_header.colorspace, self.rgb_header.colorspace)
        
        # Verify the read data matches the original
        self.assertEqual(len(read_data), len(self.rgb_data))
        for i in range(len(self.rgb_data)):
            self.assertEqual(read_data[i], self.rgb_data[i])
    
    def test_file_io_rgba(self):
        """Test writing and reading RGBA QOI files"""
        # Write the RGBA data to a file
        write(self.rgba_file, self.rgba_data, self.rgba_header, len(self.rgba_data))
        
        # Verify the file exists
        self.assertTrue(os.path.exists(self.rgba_file))
        
        # Create a new header for reading
        read_header = QoiHeader(0, 0, 0, 0)
        
        # Read the file
        read_data = read(self.rgba_file, read_header)
        
        # Verify the read header matches the original
        self.assertEqual(read_header.width, self.rgba_header.width)
        self.assertEqual(read_header.height, self.rgba_header.height)
        self.assertEqual(read_header.channels, self.rgba_header.channels)
        self.assertEqual(read_header.colorspace, self.rgba_header.colorspace)
        
        # Verify the read data matches the original
        self.assertEqual(len(read_data), len(self.rgba_data))
        for i in range(len(self.rgba_data)):
            self.assertEqual(read_data[i], self.rgba_data[i])
    
    def test_channel_conversion(self):
        """Test conversion between RGB and RGBA during decoding"""
        # Encode the RGBA data
        encoded_data, encoded_len = encode(self.rgba_data, self.rgba_header, len(self.rgba_data))
        
        # Create a new header for decoding
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # Decode as RGB (3 channels)
        decoded_rgb = decode(encoded_data, encoded_len, decode_header, channels=3)
        
        # Verify the data length is correct (2x2 pixels, 3 channels each)
        self.assertEqual(len(decoded_rgb), 2 * 2 * 3)
        
        # Verify the RGB color values match the original RGBA (without alpha)
        for i in range(0, len(decoded_rgb), 3):
            rgba_idx = (i // 3) * 4
            self.assertEqual(decoded_rgb[i], self.rgba_data[rgba_idx])
            self.assertEqual(decoded_rgb[i+1], self.rgba_data[rgba_idx+1])
            self.assertEqual(decoded_rgb[i+2], self.rgba_data[rgba_idx+2])
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        # Test with None data
        result, length = encode(None, self.rgb_header, len(self.rgb_data))
        self.assertIsNone(result)
        
        # Test with invalid header
        invalid_header = QoiHeader(0, 0, 3, 0)  # Zero dimensions
        result, length = encode(self.rgb_data, invalid_header, len(self.rgb_data))
        self.assertIsNone(result)
        
        # Test with invalid channels
        invalid_header = QoiHeader(2, 2, 2, 0)  # 2 channels (invalid)
        result, length = encode(self.rgb_data, invalid_header, len(self.rgb_data))
        self.assertIsNone(result)
        
        # Test with too many channels
        invalid_header = QoiHeader(2, 2, 5, 0)  # 5 channels (invalid)
        result, length = encode(self.rgb_data, invalid_header, len(self.rgb_data))
        self.assertIsNone(result)
        
        # Test with invalid colorspace
        invalid_header = QoiHeader(2, 2, 3, 2)  # Colorspace 2 (invalid)
        result, length = encode(self.rgb_data, invalid_header, len(self.rgb_data))
        self.assertIsNone(result)
        
        # Test decode with None data
        result = decode(None, 100, self.rgb_header)
        self.assertIsNone(result)
        
        # Test read with non-existent file
        result = read("non_existent_file.qoi", self.rgb_header)
        self.assertIsNone(result)
    
    def test_colorspace_linear(self):
        """Test encoding and decoding with linear colorspace"""
        # Create header with linear colorspace
        linear_header = QoiHeader(2, 2, 3, QOI_LINEAR)
        
        # Encode with linear colorspace
        encoded_data, encoded_len = encode(self.rgb_data, linear_header, len(self.rgb_data))
        
        # Ensure encoding produced data
        self.assertIsNotNone(encoded_data)
        
        # Create a new header for decoding
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # Decode the data
        decoded_data = decode(encoded_data, encoded_len, decode_header)
        
        # Verify the decoded header has linear colorspace
        self.assertEqual(decode_header.colorspace, QOI_LINEAR)
        
        # Verify the decoded data matches the original
        self.assertEqual(len(decoded_data), len(self.rgb_data))
        for i in range(len(self.rgb_data)):
            self.assertEqual(decoded_data[i], self.rgb_data[i])
    
    def test_large_image(self):
        """Test encoding and decoding of a larger image"""
        # Create a 100x100 RGB image with random colors
        width, height = 100, 100
        large_data = bytearray(np.random.randint(0, 256, size=(width * height * 3), dtype=np.uint8))
        large_header = QoiHeader(width=width, height=height, channels=3, colorspace=QOI_SRGB)
        
        # Encode the large data
        encoded_data, encoded_len = encode(large_data, large_header, len(large_data))
        
        # Ensure encoding produced data
        self.assertIsNotNone(encoded_data)
        
        # Create a new header for decoding
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # Decode the data
        decoded_data = decode(encoded_data, encoded_len, decode_header)
        
        # Verify the decoded header matches the original
        self.assertEqual(decode_header.width, large_header.width)
        self.assertEqual(decode_header.height, large_header.height)
        self.assertEqual(decode_header.channels, large_header.channels)
        
        # Verify the decoded data matches the original
        self.assertEqual(len(decoded_data), len(large_data))
        for i in range(len(large_data)):
            self.assertEqual(decoded_data[i], large_data[i])
    
    def test_color_hash_index(self):
        """Test the color hash and index mechanism by creating an image with repeating colors"""
        # Create an image with repeating color patterns (to test index lookups)
        colors = [
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [255, 255, 255],# White
            [0, 0, 0]       # Black
        ]
        
        # Create 8x8 image with repeating colors
        pattern_data = bytearray()
        for i in range(8):
            for j in range(8):
                color_idx = (i + j) % len(colors)
                pattern_data.extend(colors[color_idx])
        
        pattern_header = QoiHeader(width=8, height=8, channels=3, colorspace=QOI_SRGB)
        
        # Encode the pattern data
        encoded_data, encoded_len = encode(pattern_data, pattern_header, len(pattern_data))
        
        # Pattern with repeating colors should compress well
        self.assertLess(encoded_len, len(pattern_data))
        
        # Create a new header for decoding
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # Decode the data
        decoded_data = decode(encoded_data, encoded_len, decode_header)
        
        # Verify the decoded data matches the original
        self.assertEqual(len(decoded_data), len(pattern_data))
        for i in range(len(pattern_data)):
            self.assertEqual(decoded_data[i], pattern_data[i])

    def test_fuzzing(self):
        """Test the robustness of the decoder against corrupted data"""
        import random
        
        # First create valid encoded data
        encoded_data, encoded_len = encode(self.rgba_data, self.rgba_header, len(self.rgba_data))
        self.assertIsNotNone(encoded_data)
        
        # Create a mutable copy of the encoded data
        fuzzed_data = bytearray(encoded_data)
        
        # Number of bytes to corrupt (about 10% of the data)
        num_corruptions = max(1, encoded_len // 10)
        
        # Fuzzing test iterations
        for _ in range(5):  # Run 5 different corruption patterns
            # Reset fuzzed data to original
            fuzzed_data = bytearray(encoded_data)
            
            # Corrupt random bytes
            for _ in range(num_corruptions):
                # Choose a random position (avoiding header bytes for better test focus)
                pos = random.randint(14, encoded_len - 1)  # QOI header is 14 bytes
                
                # Replace with random byte
                fuzzed_data[pos] = random.randint(0, 255)
            
            # Create header for decoding
            decode_header = QoiHeader(0, 0, 0, 0)
            
            # Try to decode the corrupted data - it shouldn't crash
            try:
                decoded_data = decode(bytes(fuzzed_data), encoded_len, decode_header)
                
                # It's okay if decode returns None for badly corrupted data
                if decoded_data is not None:
                    # If we got data back, make sure it's the right size or None
                    # (The exact pixel values might not match due to corruption)
                    expected_size = self.rgba_header.width * self.rgba_header.height * self.rgba_header.channels
                    self.assertEqual(len(decoded_data), expected_size)
            except Exception as e:
                self.fail(f"Decoder crashed on corrupted data: {str(e)}")
        
        # Test totally random data
        random_data = bytearray(random.randint(0, 255) for _ in range(100))
        decode_header = QoiHeader(0, 0, 0, 0)
        
        # This should not crash, but may return None
        try:
            result = decode(bytes(random_data), len(random_data), decode_header)
            # We're not asserting anything about the result, just that it doesn't crash
        except Exception as e:
            self.fail(f"Decoder crashed on random data: {str(e)}")
if __name__ == "__main__":
    unittest.main(verbosity=2)