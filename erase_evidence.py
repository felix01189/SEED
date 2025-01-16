import json
from charset_normalizer import detect

input_file = './data/bird/dev/dev.json'
output_file = './dev_no_evi.json'

with open(input_file, 'rb') as f:
    raw_data = f.read()
    encoding = detect(raw_data)['encoding']
with open(input_file, 'r', encoding=encoding) as f:
    data = json.load(f)

for item in data:
    if 'evidence' in item:
        item['evidence'] = ""

with open(output_file, 'w', encoding=encoding) as f:
    json.dump(data, f, indent=2)