#!/usr/bin/env python3
"""
Обробка HTML файлів - просто змініть шляхи і запустіть
"""

from html_rag import create_pipeline
import glob
import os

# ========== ЗМІНІТЬ ЦІ ШЛЯХИ ==========
HTML_FOLDER = "./sample-sites"        # Папка з HTML файлами
DB_PATH = "./database"           # Шлях до бази даних
# ====================================

def main():
    print("🚀 Починаємо обробку HTML файлів...")

    # Створюємо пайплайн
    pipeline = create_pipeline(db_path=DB_PATH)

    # Знаходимо всі HTML файли
    html_files = glob.glob(os.path.join(HTML_FOLDER, "*.html"))
    html_files += glob.glob(os.path.join(HTML_FOLDER, "*.htm"))

    print(f"📁 Знайдено {len(html_files)} файлів")

    # Обробляємо кожен файл
    for i, file_path in enumerate(html_files, 1):
        print(f"[{i}/{len(html_files)}] Обробляю: {os.path.basename(file_path)}")

        try:
            # Читаємо файл
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Додаємо до бази
            result = pipeline.process_html(
                html_content,
                url=file_path
            )
            if result.get('success'):
                print(f"✅ Готово")
            else:
                print(f"❌ Помилка: {result.get('error', 'Невідома помилка')}")

        except Exception as e:
            print(f"❌ Помилка: {e}")

    print(f"🎉 Обробка завершена!")

if __name__ == "__main__":
    main()