set -e

dataset_json_path="./data/spider/test.json"
db_path="./data/spider/test_database"
output_path="./spider_test_seed_evi.json"
train_json_path="./data/bird/train/train.json"
train_db_path="./data/bird/train/train_databases"
top_k=1
openai_api_key=`cat openai_api_key`
deepseek_api_key=`cat deepseek_api_key`

python make_evidence.py \
--dataset_json_path $dataset_json_path \
--db_path $db_path \
--output_path $output_path \
--train_json_path $train_json_path \
--train_db_path $train_db_path \
--top_k $top_k \
--openai_api_key $openai_api_key \
--deepseek_api_key $deepseek_api_key
