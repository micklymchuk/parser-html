#!/usr/bin/env python3
"""
Пошук по базі даних - змініть шлях до БД і запит
"""

from html_rag import create_pipeline

# ========== ЗМІНІТЬ ЦЕ ==========
DB_PATH = "./database"           # Шлях до бази даних
SEARCH_QUERY = "знайди хто проти приватизації"      # Що шукати
# ==============================

def main():
    print(f"🔍 Шукаю: '{SEARCH_QUERY}'")
    print(f"📂 В базі: {DB_PATH}")

    # Підключаємося до бази
    pipeline = create_pipeline(db_path=DB_PATH)

    # Шукаємо (без обмежень, всі результати)
    results = pipeline.topic_aware_search(
        query=SEARCH_QUERY,
        n_results=10             # Максимум результатів
    )

    print(f"\n📋 Знайдено {len(results)} результатів:\n")

    # Виводимо все що знайшли
    for i, result in enumerate(results, 1):
        text = result['text']
        score = result['similarity_score']
        metadata = result.get('metadata', {})
        source = metadata.get('url', 'Невідоме джерело')

        print(f"--- Результат {i} ---")
        print(f"Схожість: {score:.3f}")
        print(f"Джерело: {source}")
        print(f"Текст: {text}")
        print(f"Metadata: {metadata}")
        print("-" * 50)

if __name__ == "__main__":
    main()