set -e

dataset_json_path="./dev.json"
db_path="./database"
output_path="./dev_descriptions.json"
openai_api_key=`cat openai_api_key`
deepseek_api_key=`cat deepseek_api_key`

python spider_description.py \
--dataset_json_path $dataset_json_path \
--db_path $db_path \
--output_path $output_path \
--openai_api_key $openai_api_key \
--deepseek_api_key $deepseek_api_key
