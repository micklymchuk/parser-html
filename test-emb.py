#!/usr/bin/env python3
"""
Демонстрація як працює семантичне розуміння в вашому пайплайні
"""

from sentence_transformers import SentenceTransformer
import numpy as np

def demonstrate_semantic_understanding():
    """Показує як модель 'розуміє' слова"""

    print("🧠 Завантажую NLP модель (це може зайняти хвилину)...")

    # Та сама модель що використовує ваш пайплайн
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print("✅ Модель завантажена! Вона 'розуміє' значення слів\n")

    # Тестові фрази
    test_phrases = [
        "Юлія Тимошенко",
        "лідер партії Батьківщина",
        "політик України",
        "економічні реформи",
        "приватизація підприємств",
        "соціальна політика",
        "український регіон",
        "розвиток країни",
        "яблуко червоне",
        "кішка спить"
    ]

    print("🔢 Перетворюю слова в числові вектори (embeddings):")
    print("="*70)

    # Створюємо вектори для всіх фраз
    embeddings = model.encode(test_phrases)

    for i, phrase in enumerate(test_phrases):
        vector_sample = embeddings[i][:5]  # Показуємо тільки перші 5 чисел
        print(f"{phrase:25} → [{vector_sample[0]:6.3f}, {vector_sample[1]:6.3f}, {vector_sample[2]:6.3f}, ...]")

    print(f"\n💡 Кожен вектор має {len(embeddings[0])} чисел!")

    # Демонструємо схожість
    print("\n🎯 Обчислюю семантичну схожість:")
    print("="*70)

    target_phrase = "Тимошенко Батьківщина"
    target_embedding = model.encode([target_phrase])[0]

    print(f"🎯 Запит: '{target_phrase}'")
    print("\nСхожість з іншими фразами:")

    similarities = []
    for i, phrase in enumerate(test_phrases):
        # Обчислюємо косинусну схожість
        similarity = np.dot(target_embedding, embeddings[i]) / (
                np.linalg.norm(target_embedding) * np.linalg.norm(embeddings[i])
        )
        similarities.append((phrase, similarity))

    # Сортуємо за схожістю
    similarities.sort(key=lambda x: x[1], reverse=True)

    for phrase, similarity in similarities:
        emoji = "🔥" if similarity > 0.7 else "✅" if similarity > 0.5 else "🤔" if similarity > 0.3 else "❌"
        print(f"{emoji} {similarity:.3f} - {phrase}")

    print("\n" + "="*70)
    print("💡 ПОЯСНЕННЯ:")
    print("- 1.000 = ідентичний збіг")
    print("- 0.800+ = дуже схожі за змістом")
    print("- 0.500+ = схожі тематично")
    print("- 0.300+ = слабка схожість")
    print("- 0.000- = зовсім різні")

    print("\n🎓 ОСЬ ЧОМУ ваш пошук знаходить 'українські регіони':")
    print("- Модель 'бачить' що це про українську політику")
    print("- 'Регіональна політика' схожа на 'політичні партії'")
    print("- Векторний пошук знаходить семантично близькі тексти")

if __name__ == "__main__":
    try:
        demonstrate_semantic_understanding()
    except Exception as e:
        print(f"❌ Помилка: {e}")
        print("💡 Переконайтесь що встановлено: pip install sentence-transformers")