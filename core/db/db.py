import sqlite3
import csv
from datetime import datetime


class Database:
    def __init__(self):
        self.db_name = 'experiments.db'
        self.create_database()


    def create_database(self):
        """Создает базу данных и таблицу для хранения результатов экспериментов"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            data_file_path TEXT,
            generation_params BOOLIAN,
            method1_result INTEGER,
            method2_result INTEGER,
            method3_result INTEGER
        )
        ''')

        conn.commit()
        conn.close()


    def clear_database(self):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM experiments')

        cursor.execute('UPDATE SQLITE_SEQUENCE SET seq = 0 WHERE name = "experiments"')

        conn.commit()
        conn.close()


    def add_experiment(self, data_file_path, generation_params, method1, method2, method3):
        """Добавляет новый эксперимент в базу данных"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        cursor.execute('''
        INSERT INTO experiments (date, data_file_path, generation_params, method1_result, method2_result, method3_result)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (current_date, data_file_path, generation_params, method1, method2, method3))

        conn.commit()
        conn.close()


    def export_to_csv(self, output_file='experiments.csv'):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        # Получаем все эксперименты
        cursor.execute('SELECT * FROM experiments')
        experiments = cursor.fetchall()

        # Получаем названия столбцов
        cursor.execute('PRAGMA table_info(experiments)')
        columns = [column[1] for column in cursor.fetchall()]

        conn.close()

        # Записываем в CSV файл
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Записываем заголовки
            writer.writerow([
                'Номер эксперимента',
                'Дата',
                'Путь к файлу',
                'Сгенерированные данные (1 - да, 0 - нет)',
                'CV (количество клеток)',
                'ML (количество клеток)',
                'CNN (количество клеток)'
            ])

            # Записываем данные
            for exp in experiments:
                writer.writerow(exp)

        print(f'Данные успешно экспортированы в {output_file}')


    def get_experiments_by_ids(self, experiment_ids):
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        placeholders = ','.join(['?'] * len(experiment_ids))
        query = f'SELECT * FROM experiments WHERE experiment_id IN ({placeholders})'

        cursor.execute(query, experiment_ids)
        experiments = cursor.fetchall()

        conn.close()
        return experiments




# Пример использования
if __name__ == '__main__':
    # Создаем базу данных (если не существует)
    db = Database()

    # Добавляем тестовые данные
    db.add_experiment(
        data_file_path='data/experiment1.csv',
        generation_params=0,
        method1=512,
        method2=498,
        method3=505
    )

    db.add_experiment(
        data_file_path='data/experiment2.csv',
        generation_params=1,
        method1=1023,
        method2=998,
        method3=1010
    )

    db.export_to_csv()

    experiments = db.get_experiments_by_ids([1, 2])
    for exp in experiments:
        print(exp)