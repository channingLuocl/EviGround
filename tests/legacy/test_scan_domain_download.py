"""Resume must append exactly the advertised byte range."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from eviground.experiments.download_scan_domain_holdout import acquire


class Response(io.BytesIO):
    def __init__(self, payload, status, headers):
        super().__init__(payload)
        self.status, self.headers = status, headers


class DownloadTests(unittest.TestCase):
    def test_resume_preserves_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); target=root/'scene0001_00/scene0001_00.txt'
            target.parent.mkdir(); part=target.with_suffix('.txt.partial'); part.write_bytes(b'abc')
            response=Response(b'def',206,{'Content-Range':'bytes 3-5/6','Content-Length':'3'})
            with patch('urllib.request.urlopen',return_value=response) as request:
                self.assertEqual(acquire(('scene0001_00','.txt',root)),3)
                self.assertEqual(request.call_args.args[0].headers['Range'],'bytes=3-')
            self.assertEqual(target.read_bytes(),b'abcdef')

    def test_ignored_range_cannot_corrupt_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); target=root/'scene0001_00/scene0001_00.txt'
            target.parent.mkdir(); part=target.with_suffix('.txt.partial'); part.write_bytes(b'abc')
            with patch('urllib.request.urlopen',return_value=Response(b'abcdef',200,{})):
                with self.assertRaises(ValueError): acquire(('scene0001_00','.txt',root))
            self.assertFalse(target.exists())
            self.assertEqual(part.read_bytes(),b'abc')


if __name__=='__main__': unittest.main()
