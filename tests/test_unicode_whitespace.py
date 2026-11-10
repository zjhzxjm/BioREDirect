"""Exercise the frozen GAS caption with the real BERT tokenizer on ai-0."""
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/dataset_format_converter'))
from annotation import AnnotationInfo
from document import Document, TextInstance
from transformers import BertTokenizer
from utils import tokenize_document_by_bert


class UnicodeWhitespaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = BertTokenizer.from_pretrained(os.environ['BIORED_TEST_MODEL'], local_files_only=True)

    def test_caption_and_mention_spaces_preserve_offsets(self):
        for gap in (' ', '\u2002\u2009', '\u00a0', '\t'):
            for mention in ('GAS', 'chemical' + gap + 'structure'):
                text = 'Fig. 1 ' + gap + ' The chemical' + gap + 'structure of gastrodin (GAS)'
                with self.subTest(gap=repr(gap), mention=mention):
                    document = Document('caption')
                    instance = TextInstance(text)
                    entity = AnnotationInfo(text.index(mention), len(mention), mention, 'ChemicalEntity')
                    instance.annotations.append(entity)
                    document.text_instances.append(instance)
                    tokenize_document_by_bert(document, self.tokenizer)
                    self.assertEqual(instance.tokens[entity.start_token:entity.end_token + 1],
                                     self.tokenizer.tokenize(mention))
                    self.assertEqual(instance.text, text)
                    self.assertEqual(text[entity.position:entity.position + entity.length], mention)


if __name__ == '__main__':
    unittest.main()
