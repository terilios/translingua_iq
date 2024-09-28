import os

def generate_test_files():
    english_source = "The quick brown fox jumps over the lazy dog."
    translations = {
        "es": "El rápido zorro marrón salta sobre el perro perezoso.",
        "fr": "Le rapide renard brun saute par-dessus le chien paresseux.",
        "de": "Der schnelle braune Fuchs springt über den faulen Hund.",
        "it": "La rapida volpe marrone salta sopra il cane pigro.",
        "pt": "A rápida raposa marrom salta sobre o cão preguiçoso.",
        "nl": "De snelle bruine vos springt over de luie hond.",
        "ru": "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "zh": "快速的棕色狐狸跳过懒惰的狗。",
        "ja": "素早い茶色のキツネが怠け者の犬を飛び越える。"
    }

    if not os.path.exists("test_files"):
        os.makedirs("test_files")

    with open("test_files/english_source.txt", "w", encoding="utf-8") as f:
        f.write(english_source)

    for lang, translation in translations.items():
        with open(f"test_files/{lang}_translation.txt", "w", encoding="utf-8") as f:
            f.write(translation)

    print("Test files generated successfully.")

if __name__ == "__main__":
    generate_test_files()