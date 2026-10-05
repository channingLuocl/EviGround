import unittest
from types import SimpleNamespace
from unittest.mock import patch
from qwen_api import chat, QwenAPIError


class QwenAPITests(unittest.TestCase):
    def test_request_and_finish_reason(self):
        response=SimpleNamespace(ok=True,json=lambda:{'choices':[{'finish_reason':'length',
                                  'message':{'content':'Predicted ID: 1'}}],'usage':{'total_tokens':12}})
        with patch.dict('os.environ',{'SEEGROUND_MODEL':'mock','SEEGROUND_BASE_URL':'https://example.invalid/v1',
                                    'SEEGROUND_API_KEY':'test-only'}), \
             patch('qwen_api.requests.post',return_value=response) as post:
            result=chat([{'role':'user','content':'test'}],max_tokens=768)
        self.assertEqual(post.call_args.args[0],'https://example.invalid/v1/chat/completions')
        self.assertEqual(post.call_args.kwargs['json']['max_tokens'],768)
        self.assertEqual(result['_finish_reason'],'length')
        self.assertEqual(result['_content'],'Predicted ID: 1')

    def test_bad_response_is_value_error(self):
        with patch.dict('os.environ',{'SEEGROUND_MODEL':'mock','SEEGROUND_BASE_URL':'https://example.invalid/v1',
                                    'SEEGROUND_API_KEY':'test-only'}), \
             patch('qwen_api.requests.post',return_value=SimpleNamespace(ok=True,json=lambda:{})):
            with self.assertRaises(QwenAPIError):
                chat([])


if __name__=='__main__':unittest.main()
