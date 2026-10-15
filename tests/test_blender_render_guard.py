import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import blender_render_guard

class TestBlenderRenderGuard(unittest.TestCase):

    def test_partition_frames(self):
        # Even split
        chunks = blender_render_guard.partition_frames(1, 4, 2)
        self.assertEqual(chunks, [(1, 2), (3, 4)])

        # Uneven split
        chunks = blender_render_guard.partition_frames(1, 3, 2)
        self.assertEqual(chunks, [(1, 2), (3, 3)])

        # More chunks than frames
        chunks = blender_render_guard.partition_frames(1, 2, 4)
        self.assertEqual(chunks, [(1, 1), (2, 2)])

        # Single chunk
        chunks = blender_render_guard.partition_frames(1, 10, 1)
        self.assertEqual(chunks, [(1, 10)])

        # Invalid range
        chunks = blender_render_guard.partition_frames(5, 1, 2)
        self.assertEqual(chunks, [])

    @patch('subprocess.run')
    def test_get_cuda_devices_success(self, mock_run):
        mock_result = MagicMock()
        mock_result.stdout = "0\n1\n"
        mock_run.return_value = mock_result

        devices = blender_render_guard.get_cuda_devices()
        self.assertEqual(devices, ['0', '1'])

    @patch('subprocess.run')
    def test_get_cuda_devices_fallback(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        devices = blender_render_guard.get_cuda_devices()
        self.assertEqual(devices, ['0'])

    def test_verify_image_png_success(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b'\x89PNG\r\n\x1a\n') # PNG header
            f.write(b'dummydata')
            temp_path = f.name

        self.assertTrue(blender_render_guard.verify_image(temp_path))
        os.remove(temp_path)

    def test_verify_image_exr_success(self):
        with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as f:
            f.write(b'\x76\x2f\x31\x01') # EXR header
            f.write(b'dummydata')
            temp_path = f.name

        self.assertTrue(blender_render_guard.verify_image(temp_path))
        os.remove(temp_path)

    def test_verify_image_empty(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name

        self.assertFalse(blender_render_guard.verify_image(temp_path))
        os.remove(temp_path)

    def test_verify_image_invalid_header(self):
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b'invalidheader')
            f.write(b'dummydata')
            temp_path = f.name

        self.assertFalse(blender_render_guard.verify_image(temp_path))
        os.remove(temp_path)

if __name__ == '__main__':
    unittest.main()
