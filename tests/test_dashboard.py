import asyncio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi import HTTPException
from server.app import get_run_files, preview_file, workspace_path


class TestWorkspacePreview(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cwd = patch('os.getcwd', return_value=self.temp.name)
        self.cwd.start()
        self.run = Path(self.temp.name) / 'workspaces' / 'saved_run'
        (self.run / 'pages').mkdir(parents=True)
        (self.run / 'pages' / 'calculator.html').write_text('<h1>Calculator</h1>')
        (self.run / 'pages' / 'empty.js').write_text('')

    def tearDown(self):
        self.cwd.stop()
        self.temp.cleanup()

    def test_recursive_files_and_read_errors(self):
        (self.run / 'image.bin').write_bytes(b'\xff\xfe')
        (self.run / 'large.txt').write_text('x' * (1024 * 1024 + 1))
        result = asyncio.run(get_run_files('saved_run'))
        self.assertIn('pages/calculator.html', result['files'])
        self.assertEqual(result['files']['pages/empty.js'], '')
        self.assertIn('image.bin', result['errors'])
        self.assertIn('large.txt', result['errors'])

    def test_traversal_and_symlink_rejected(self):
        outside = Path(self.temp.name) / 'secret.txt'
        outside.write_text('secret')
        (self.run / 'link.txt').symlink_to(outside)
        for run_id, filename in [('saved_run', '../../secret.txt'), ('..', ''), ('saved_run', str(outside)), ('saved_run', 'link.txt')]:
            with self.subTest(run_id=run_id, filename=filename):
                with self.assertRaises(HTTPException):
                    workspace_path(run_id, filename)
        result = asyncio.run(get_run_files('saved_run'))
        self.assertNotIn('link.txt', result['files'])
        self.assertIn('link.txt', result['errors'])

    def test_non_index_preview_is_isolated(self):
        response = asyncio.run(preview_file('saved_run', 'pages/calculator.html'))
        self.assertEqual(response.media_type, 'text/html')
        self.assertIn('sandbox allow-scripts', response.headers['content-security-policy'])
        self.assertNotIn('allow-same-origin', response.headers['content-security-policy'])
        with self.assertRaises(HTTPException) as failure:
            asyncio.run(preview_file('saved_run', 'missing.html'))
        self.assertEqual(failure.exception.status_code, 404)
