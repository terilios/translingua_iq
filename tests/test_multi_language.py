import unittest
from app import detect_language, calculate_meteor_score, compare_named_entities, calculate_semantic_similarity, calculate_lexical_similarity

class TestMultiLanguageSupport(unittest.TestCase):

    def test_language_detection(self):
        self.assertEqual(detect_language("Hello, world!"), "en")
        self.assertEqual(detect_language("Bonjour le monde!"), "fr")
        self.assertEqual(detect_language("Hola mundo!"), "es")

    def test_meteor_score(self):
        score = calculate_meteor_score("This is a test.", "This is a test.", "en")
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_named_entity_comparison(self):
        source = "Barack Obama was the president of the United States."
        translation = "Barack Obama fue el presidente de los Estados Unidos."
        score = compare_named_entities(source, translation, "en", "es")
        self.assertGreater(score, 0.5)

    def test_semantic_similarity(self):
        source = "The cat is on the mat."
        translation = "El gato está sobre la alfombra."
        score = calculate_semantic_similarity(source, translation)
        self.assertGreater(score, 0.5)

    def test_lexical_similarity(self):
        source = "Hello, world!"
        translation = "Hola, mundo!"
        score = calculate_lexical_similarity(source, translation)
        self.assertGreater(score, 0.5)

if __name__ == '__main__':
    unittest.main()