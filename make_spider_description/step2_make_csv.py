import os
import sqlite3
import csv

def export_db_to_csv(dev_database_dir):
    for db_name in os.listdir(dev_database_dir):
        db_path = os.path.join(dev_database_dir, db_name, f"{db_name}.sqlite")
        if os.path.isfile(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
            tables = [row[0] for row in cursor.fetchall()]
            
            description_dir = os.path.join(dev_database_dir, db_name, "database_description")
            os.makedirs(description_dir, exist_ok=True)
            
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                columns = [(row[1], row[2]) for row in cursor.fetchall()]

                cursor.execute("PRAGMA encoding;")
                encoding = [str(row[0]) for row in cursor.fetchall()][0]

                csv_path = os.path.join(description_dir, f"{table}.csv")
                with open(csv_path, "w", newline="", encoding=encoding) as f:
                    writer = csv.writer(f)
                    writer.writerow(["original_column_name", "column_name", "column_description", "data_format", "value_description"])
                    
                    for col_name, col_type in columns:
                        cursor.execute(f"SELECT DISTINCT `{col_name}` FROM `{table}` LIMIT 5;")
                        distinct_values = [str(row[0]) for row in cursor.fetchall()]
                        value_description = ", ".join(distinct_values)
                        writer.writerow([col_name, col_name, "", col_type, value_description])
                    
                print(f"Exported {table} to {csv_path}")
            
            conn.close()

if __name__ == "__main__":
    database_dir = "database" 
    export_db_to_csv(database_dir)

