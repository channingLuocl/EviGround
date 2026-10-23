import struct
import tempfile
import unittest
from pathlib import Path

from run_facing_anchor_pose_experiment import mesh_vertices


class MeshVerticesTest(unittest.TestCase):
    def test_reads_binary_vertex_records_once(self):
        header = b"".join([
            b"ply\n", b"format binary_little_endian 1.0\n", b"element vertex 2\n",
            b"property float x\n", b"property float y\n", b"property float z\n",
            b"property uchar red\n", b"end_header\n",
        ])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.ply"
            path.write_bytes(header + struct.pack("<fffB", 1.0, 2.0, 3.0, 4) + struct.pack("<fffB", 5.0, 6.0, 7.0, 8))
            self.assertEqual(mesh_vertices(path), [(1.0, 2.0, 3.0), (5.0, 6.0, 7.0)])


if __name__ == "__main__":
    unittest.main()
