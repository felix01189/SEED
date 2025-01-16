import json
import argparse
from openai import OpenAI
import sqlite3
import time
import re
from charset_normalizer import detect


###settting#####################################################################################
# gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20, deepseek-chat
temperature = 0.
llm_model="deepseek-chat"
################################################################################################

def parse_option():
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("--dataset_json_path", type=str)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--openai_api_key", type=str, default="")
    parser.add_argument("--deepseek_api_key", type=str, default="")

    opt = parser.parse_args()

    return opt

def make_prompt(db_id, table_name, concat_schema):
    system_prompt = f"""### You are a data science expert and should assist your colleague in creating SQL. \
Interpret the db schema given below and explain the meaning of each column of table `{table_name}` in one sentence. \
This information will be used to assist text-to-SQL operations. \
Provide the output in the following unannotated JSON format:
	{{
        "db_id": "{db_id}"
        "db_id": "{table_name}"
        "reaoning": "Describe the reasoning step in detail"
	    "description": "Provide clear, concise and accurate evidence"
	}}
"""

    user_prompt = f"""### schema ####################################################
{{
    {concat_schema}}}

### Let's think step by step.
"""

    return system_prompt, user_prompt


def generate_reply(input, model, temperature):

    if 'gpt' in model:
        client = OpenAI(api_key=opt.openai_api_key)

        response = client.chat.completions.create(
            model=model,
            messages=input,
            temperature=temperature,
            stream=False
        )

        return response.choices[0].message.content
    
    else: # deepseek
        client = OpenAI(api_key=opt.deepseek_api_key, base_url = "https://api.deepseek.com/v1")

        response = client.chat.completions.create(
            model=model,
            messages=input,
            temperature=temperature,
            stream=False
        )

        return response.choices[0].message.content


def get_table_names(db_id):
    db_path=f"{opt.db_path}/{db_id}/{db_id}.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT tbl_name FROM sqlite_master WHERE type='table' order by tbl_name;")
    tables = [row[0] for row in cursor.fetchall()]
    return tables


def generate_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' order by tbl_name;")
    schemas = cursor.fetchall()
    schema = ""
    for sc in schemas:
        schema += (sc[0] + "\n\n")
    return schema


def read_schema_description(db_path, num_of_sampling=30):
    schema_description = ""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    sql = "SELECT m.name AS table_name, p.name AS column_name FROM sqlite_master AS m JOIN pragma_table_info(m.name) AS p WHERE m.type = 'table' ORDER BY m.name ASC, p.cid ASC;"
    cursor.execute(sql)
    results = cursor.fetchall()
    
    for table_name, column_name in results:
        try:
            sql = f"select lower(type) from pragma_table_info('{table_name}') where name='{column_name}';"
            cursor.execute(sql)
            column_values = cursor.fetchall()
            if "blob" in column_values:
                schema_description += '\n'
            else:
                sql = f"SELECT distinct case when substr(`{column_name}`,1,100) is null then 'null' else substr(`{column_name}`,1,100) end FROM (SELECT `{column_name}` FROM `{table_name}` limit 1000) limit {num_of_sampling};"
                cursor.execute(sql)
                column_values = cursor.fetchall()
                column_value = ""
                for value in column_values:
                    column_value += (str(value[0]).replace('\n', ' ') + ", ")
                schema_description = schema_description + "   ### column value examples: " + column_value + '\n'
        except Exception as e:
            schema_description += '\n'
    
    conn.close()
    return schema_description


def concat_schema_and_desc(schema, schema_description):
    if schema_description == "":
        return schema

    try:
        schema_lines = schema.splitlines()
        schema_description_lines = schema_description.splitlines()

        concat_schema_lines = []
        schema_description_index = 0

        for schema_line in schema_lines:
            if schema_line \
            and schema_line.strip() != ""\
            and not schema_line.startswith("CREATE") \
            and not schema_line.startswith("--") \
            and not schema_line.startswith(")") \
            and not schema_line.startswith("(") \
            and not schema_line.strip().lower().startswith("unique") \
            and not schema_line.strip().lower().startswith("references") \
            and not schema_line.strip().lower().startswith("on update cascade") \
            and not schema_line.strip().lower().startswith("primary key") \
            and not schema_line.strip().lower().startswith("constraint") \
            and not schema_line.strip().lower().startswith("foreign key") \
            and not schema_line.strip().lower().startswith("unique"):
                concat_schema_lines.append(schema_line + schema_description_lines[schema_description_index])
                schema_description_index += 1
            else:
                concat_schema_lines.append(schema_line)

        concat_schema = "\n".join(concat_schema_lines)
        return concat_schema

    except Exception as e:
        print(f"Warning) Error schema concat : {e}")
        return schema



def description_generation(db_id, table_name):

    schema = generate_schema(f"{opt.db_path}/{db_id}/{db_id}.sqlite")
    schema_description = read_schema_description(f"{opt.db_path}/{db_id}/{db_id}.sqlite",30)
    concat_schema = concat_schema_and_desc(schema, schema_description)

    system_prompt, user_prompt = make_prompt(db_id, table_name, concat_schema,)

    prompt = [{"role": "system", "content": system_prompt}]
    prompt.append({"role": "user", "content": user_prompt})

    # print(prompt)

    response = None
    while response is None:
        try:
            response = generate_reply(prompt, llm_model, temperature)
        except Exception as e:
            print('api error, wait for 3 seconds and retry...')
            print(e)
            time.sleep(3)

    print(response)

    description = extract_from_json(response, "description")
    return description


def extract_from_json(response_content, item):
    try:
        response_json = json.loads(response_content)
        return response_json.get(item, f"Warning) No {item} found")
    except json.JSONDecodeError:
        json_pattern = r"```json(.*?)```"
        matches = re.findall(json_pattern, response_content, re.DOTALL)
        try:
            response_json = json.loads(matches[-1])
            return response_json.get(item, f"Warning) No {item} found")
        except:
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            matches = re.findall(json_pattern, response_content, re.DOTALL)
            try:
                response_json = json.loads(matches[-1])
                return response_json.get(item, f"Warning) No {item} found")
            except:
                return "Warning) Response is not in JSON format"


if __name__ == "__main__":
    opt = parse_option()
    print(opt)
    res = []

    with open(opt.dataset_json_path, 'rb') as f:
        raw_data = f.read()
        encoding = detect(raw_data)['encoding']
    with open(opt.dataset_json_path, encoding=encoding) as f:
        question_json_all = json.load(f)
    
    db_ids = {item["db_id"] for item in question_json_all}

    for db_id in db_ids:
        table_names = get_table_names(db_id)
        for table_name in table_names:
            description = description_generation(db_id, table_name)

            data = {
                "db_id": db_id,
                "table": table_name,
                "description": description
            }

            res.append(data)
        
    with open(opt.output_path, 'w', encoding=encoding) as f:
        json.dump(res, f, indent=2)
