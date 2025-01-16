import json
import os
import shutil

def get_unique_db_ids(dev_json_path):
    with open(dev_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    unique_db_ids = {item["db_id"] for item in data}
    return unique_db_ids

def clean_unused_databases(dev_json_path, database_dir):
    unique_db_ids = get_unique_db_ids(dev_json_path)
    
    for db_name in os.listdir(database_dir):
        db_path = os.path.join(database_dir, db_name)
        if os.path.isdir(db_path) and db_name not in unique_db_ids:
            shutil.rmtree(db_path)
            print(f"Removed: {db_path}")

if __name__ == "__main__":
    dev_json_path = "dev.json" 
    database_dir = "database"
    
    clean_unused_databases(dev_json_path, database_dir)

