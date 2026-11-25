import builtins
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from text_normalization import _normalizer, normalize_text


class TextNormalizationTests(unittest.TestCase):
    def test_default_normalizes_chinese(self):
        self.assertEqual(normalize_text('12元'), '十二元')

    def test_unsupported_language_does_not_create_cache(self):
        with tempfile.TemporaryDirectory() as tmp, patch('text_normalization._normalizer') as loader:
            cache = Path(tmp)/'unused'
            self.assertEqual(normalize_text('  12 mele！\n', lang='Italian', cache_dir=cache), '  12 mele！\n')
            self.assertFalse(cache.exists())
            loader.assert_not_called()

    def test_dates_and_times(self):
        value = normalize_text('今天是2026年11月11日，下午3:30开会。')
        self.assertEqual(value, '今天是二零二六年十一月十一日，下午三点三十分开会。')

    def test_money_and_discount(self):
        self.assertEqual(normalize_text('这件衣服99.5元，打8折。'),
                         '这件衣服九十九点五元，打八折。')

    def test_percent_and_temperature(self):
        self.assertEqual(normalize_text('增长率为12.5%，温度是-3℃。'),
                         '增长率为百分之十二点五，温度是负三摄氏度。')

    def test_units(self):
        self.assertEqual(normalize_text('长度是3.5km，重量为2kg。'),
                         '长度是三点五公里，重量为两千克。')

    def test_interjections_erhua_punctuation_and_traditional_preserved(self):
        for value in ['嗯，花儿真漂亮！', '請到這裡來。']:
            self.assertEqual(normalize_text(value), value)

    def test_fullwidth_numbers(self):
        self.assertEqual(normalize_text('价格是９９元。'), '价格是九十九元。')

    def test_english_numbers(self):
        self.assertEqual(normalize_text('I have 12 apples.', lang='English'), 'I have twelve apples.')

    def test_english_money(self):
        result = normalize_text('I paid $12.50 for 3 books.', lang='en')
        self.assertEqual(result, 'I paid twelve point five dollars for three books.')

    def test_english_percent(self):
        self.assertEqual(normalize_text('The price increased by 12.5%.', lang='English'),
                         'The price increased by twelve point five percent.')

    def test_language_aliases(self):
        for code, name, text in [('zh', 'Chinese', '2元'), ('en', 'English', '12 apples'),
                                 ('ja', 'Japanese', '12個'), ('ko', 'Korean', '12000원'),
                                 ('de', 'German', '12 Äpfel'), ('fr', 'French', '12 pommes')]:
            with self.subTest(lang=name):
                self.assertEqual(normalize_text(text, lang=code), normalize_text(text, lang=name))
                self.assertEqual(normalize_text(text, lang=' ' + name.upper() + ' '),
                                 normalize_text(text, lang=code))

    def test_japanese_numbers_dates_and_punctuation(self):
        cases = [('りんごを12個買いました。', 'りんごを十二個買いました。'),
                 ('今日は2026年11月12日です。', '今日は二千二十六年十一月十二日です。'),
                 ('価格は１５００円です。', '価格は千五百円です。'),
                 ('午後3:30に出発します。', '午後三時三十分に出発します。'),
                 ('こんにちは！元気ですか？', 'こんにちは！元気ですか？')]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(normalize_text(text, lang='Japanese'), expected)

    def test_korean_money_and_time(self):
        self.assertEqual(normalize_text('가격은 12000원입니다.', lang='Korean'), '가격은 만이천 원입니다.')
        self.assertEqual(normalize_text('오후 3시 30분에 만나요.', lang='Korean'), '오후 세시 삼십분 에 만나요.')

    def test_german_numbers(self):
        self.assertEqual(normalize_text('Ich habe 12 Äpfel gekauft.', lang='German'),
                         'Ich habe zwölf Äpfel gekauft.')

    def test_french_numbers_dates_and_percent(self):
        cases = [('J’ai acheté 12 pommes.', 'J’ai acheté douze pommes.'),
                 ('Nous sommes le 12/11/2026.', 'Nous sommes le douze novembre deux mille vingt-six.'),
                 ('Le taux est de 12,5 %.', 'Le taux est de douze virgule cinq pour cent.')]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(normalize_text(text, lang='French'), expected)

    def test_multilingual_cache_is_reused_and_separate(self):
        for code in ('zh', 'en', 'ja', 'ko', 'de', 'fr'):
            normalize_text('12', lang=code)
        before = _normalizer.cache_info()
        for code in ('zh', 'en', 'ja', 'ko', 'de', 'fr'):
            normalize_text('13', lang=code)
        after = _normalizer.cache_info()
        self.assertEqual(before.misses, after.misses)
        self.assertEqual(after.hits - before.hits, 6)

    def test_sichuan_uses_chinese_tn(self):
        for text in ['这件衣服99.5元，打8折。', '今天是2026年11月12日，下午3:30开会。',
                     '增长率为12.5%，温度是-3℃。', '长度是3.5km，重量为2kg。']:
            with self.subTest(text=text):
                self.assertEqual(normalize_text(text, accent='chuan'), normalize_text(text))
        self.assertEqual(normalize_text('幺儿，买2个包子噻！', accent='chuan'), '幺儿，买两个包子噻！')

    def test_cantonese_normalizes_then_converts_to_traditional(self):
        self.assertEqual(normalize_text('这件衣服99.5元，打8折。', accent='yue'),
                         '這件衣服九十九點五元，打八折。')
        self.assertEqual(normalize_text('温度是-3℃。', accent='yue'), '温度是負三攝氏度。')
        self.assertEqual(normalize_text('呢件衫12元。', accent='yue'), '呢件衫十二元。')
        self.assertEqual(normalize_text('呢件衫12元。', lang='zh', accent=' YUE '), '呢件衫十二元。')

    def test_cantonese_conversion_receives_normalized_text(self):
        with patch('text_normalization._normalizer') as loader, \
             patch('text_normalization._traditional_converter') as converter:
            loader.return_value.normalize.return_value = '十二个'
            converter.return_value.convert.return_value = '十二個'
            self.assertEqual(normalize_text('12个', accent='yue'), '十二個')
            loader.return_value.normalize.assert_called_once_with('12个')
            converter.return_value.convert.assert_called_once_with('十二个')

    def test_non_cantonese_does_not_load_converter(self):
        with patch('text_normalization._traditional_converter') as converter:
            self.assertEqual(normalize_text('买12个'), '买十二个')
            self.assertEqual(normalize_text('买12个', accent='chuan'), '买十二个')
            self.assertEqual(normalize_text('12 apples', lang='English'), 'twelve apples')
            converter.assert_not_called()

    def test_unsupported_languages_and_dialects_unchanged(self):
        with patch('text_normalization._normalizer') as loader:
            for lang, accent in [('Italian', None), ('Japanese', 'chuan'), ('Korean', 'yue'), ('German', 'unknown'),
                                 ('Chinese', 'unknown'), ('English', 'chuan'), ('English', 'yue'),
                                 ('yue', None), (None, None)]:
                with self.subTest(lang=lang, accent=accent):
                    text = '12元 / 12個 / 12 books'
                    self.assertEqual(normalize_text(text, lang=lang, accent=accent), text)
            loader.assert_not_called()

    def test_invalid_inputs_and_reserved_tokens_rejected(self):
        for text in ['', ' ', None, '<|lang_begin|>Chinese', '<|speaker_start|>1']:
            with self.assertRaises(ValueError):
                normalize_text(text)

    def test_invalid_normalized_text_is_rejected(self):
        with patch('text_normalization._normalizer') as loader:
            for value in ['', ' ', None]:
                loader.return_value.normalize.return_value = value
                with self.subTest(value=value), self.assertRaises(ValueError):
                    normalize_text('12元')

    def test_missing_dependency_is_actionable(self):
        import_fn = builtins.__import__
        def no_tn(name, *args, **kwargs):
            if name == 'tn' or name.startswith('tn.'):
                raise ModuleNotFoundError('No module named tn')
            return import_fn(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch('builtins.__import__', side_effect=no_tn), \
             patch('text_normalization.import_module', side_effect=ModuleNotFoundError('No module named tn')):
            for lang in ('Chinese', 'English', 'Japanese', 'Korean', 'German', 'French'):
                with self.subTest(lang=lang), self.assertRaisesRegex(ImportError, 'WeTextProcessing==1.3.0'):
                    normalize_text('12', lang=lang, cache_dir=tmp)

    def test_reuses_normalizer_for_same_language_and_cache(self):
        normalize_text('3元')
        before = _normalizer.cache_info()
        normalize_text('5元')
        after = _normalizer.cache_info()
        self.assertEqual(before.misses, after.misses)
        self.assertGreater(after.hits, before.hits)

    def test_concurrent_requests(self):
        with ThreadPoolExecutor(max_workers=3) as pool:
            outputs = list(pool.map(lambda _: normalize_text('12元'), range(6)))
        self.assertEqual(outputs, ['十二元']*6)


if __name__ == '__main__':
    unittest.main()
