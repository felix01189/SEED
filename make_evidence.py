import json
import argparse
from openai import OpenAI
from tqdm import tqdm
import sqlite3
import os
import sys
import time
import csv
from sentence_transformers import SentenceTransformer
import torch
import re
from datetime import datetime
import jellyfish
from charset_normalizer import detect
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

###settting#####################################################################################
# gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20, deepseek-reasoner
keyword_extract_temperature = 0.
evidence_generation_temperature = 0.
schema_summary_temperature = 0.
keyword_extract_llm, evidence_generation_llm, schema_summary_llm = "deepseek-reasoner", "deepseek-reasoner", "deepseek-reasoner"
# keyword_extract_llm, evidence_generation_llm, schema_summary_llm = "gpt-4o-mini-2024-07-18", "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18"
embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
parallel_process = 32
schema_summary = False if "gpt" in evidence_generation_llm else True
debug_print = False
################################################################################################

def parse_option():
    parser = argparse.ArgumentParser("command line arguments")
    parser.add_argument("--dataset_json_path", type=str)
    parser.add_argument("--db_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--train_json_path", type=str,  default="./data/bird/train/train.json")
    parser.add_argument("--top_k", type=int,  default=3)
    parser.add_argument("--train_db_path", type=str,  default="./data/bird/train/train_databases")
    parser.add_argument("--openai_api_key", type=str, default="")
    parser.add_argument("--deepseek_api_key", type=str, default="")

    opt = parser.parse_args()

    return opt

def make_prompt(question, concat_schema):
    evidence_generation_system_prompt = """### You are a data science expert and should assist your colleague in creating SQL. \
Your colleague is an expert in SQL, but he does not have domain knowledge. \
So you should analyze the given question and create a clear, concise and accurate evidence to help your colleague create SQL. \
Perform the steps below to create an evidence and describe in detail the reasoning of each step.

Step-by-Step Instructions:
1. **Refer to Sample SQL results**: Your another colleague generated sample SQL for words considered keyword in question and executed it in database. Refer to the value and format. Note that your colleague may have chosen the wrong schema.
2. **Analyze the Question and Schema**: Identify key elements in the question that need mapping (columns, tables, values). Clarify ambiguities by referencing the database schema.
3. **Analyze the Few-shot samples**: Read the samples and understand the relationship between question and evidence and database schema and descriptions.
4. **Generate Evidence**: Based on what you analyzed earlier, generate evidence so that it is as short as possible and contains as much information as possible.
5. **Consideration of cautions**: Make sure that the generated evidence does not violate the cautions below.
    Cautions 1. Schema-Specific Language: Use precise terminology from the database schema to avoid ambiguity. 
    Cautions 2. Schema Formatting: When mentioning a column, mention the table containing that column together. Use the form (`table`.`column`) to refer to columns.
    Cautions 3. Case Sensitivity: Reflect the exact case of database values in your evidence to prevent mismatches. Refer to db value for accurate case utilization.
6. **Output Reasoning**: Describe the reasoning of each step in detail and print it out.
7. **Output Evidence**: Provide the output in the following unannotated JSON format:
	{
	    "evidence": "Provide clear, concise and accurate evidence"
	}
"""

    evidence_generation_user_prompt = f"""### problem ####################################################
1. schema of question
{{
    {concat_schema}}}
    
2. question
{{
    "question": "{question}",
    "evidence": 
}}

### Let's think step by step.
"""

    keyword_extract_system_prompt = """### As a data science expert, you need to perform preliminary tasks before the text-to-SQL process. \
To assist in this task, you will extract the schema and values from the given question to generate sample SQL queries. \
The goal is to verify the database structure and content by extracting a diverse set of plausible schema-value combinations. \
Since this process is exploratory, adopt a lenient and comprehensive approach that accounts for potential ambiguities or multiple interpretations. \
For example, include all plausible schemas and values, even if they are uncertain or overlapping. \
This will maximize the potential for finding relevant information during the text-to-SQL task.

Please follow these steps to extract the keywords and provide a detailed explanation for each step:
### Steps for Extraction:
1. **Problem Analysis**: Analyze the question, using database schema and domain knowledge to identify both explicitly mentioned and implied relationships.
2. **Keyword Detection (Direct Schema)**: Identify keywords or phrases in the question that directly correspond to schema names.
3. **Keyword Detection (Similar Schema)**: Search for schema names similar to the detected schema, considering name similarity, domain relevance, or data type alignment.
4. **Keyword Detection (Values)**: Identify any values (categories, conditions, strings) explicitly or implicitly mentioned in the question. Ensure even implied conditions are captured.
5. **Schema-Value Pairing**: Combine detected schemas and values into all possible combinations of `{schema, value}` pairs. Include combinations where no value is directly specified, but the schema is still relevant.
6. **Mapping to Tables**: Map columns to their respective tables, explicitly formatting them as `table.column`. In case of ambiguity, include all plausible mappings.
7. **Output Reasoning**: Describe the reasoning behind each step in detail, providing insights into how combinations were derived.
8. **Output Generation**: Present results in the following JSON format:
    {
        "schema-value-pair": [
            {"schema": "<table.column>", "value": "<value>"},
            {"schema": "<table.column>", "value": null}
        ]
    }
"""

    keyword_extract_user_prompt = f"""### problem ####################################################
1. schema of question
{{
    {concat_schema}}}
    
2. question
{{
    "question": "{question}"
    "schema-value-pair":
}}

### Let's think step by step.
"""

    return evidence_generation_system_prompt, evidence_generation_user_prompt, keyword_extract_system_prompt, keyword_extract_user_prompt



def make_schema_summary_prompt(question, concat_schema):
    system_prompt = """### As a data science expert, your task is to prepare a schema for efficient text-to-SQL operations through schema linking. \
The given schema includes comments for each column, providing descriptions and value samples. \
Identify and remove columns that are irrelevant to the provided question. \
However, ensure that columns designated as primary keys or foreign keys are preserved. \
For all remaining columns, retain their associated comments, including descriptions and value samples.

Present results in the following JSON format without description:
    {
        "summarized_schema": 
    }
"""

    user_prompt = f"""#######################################################
1. schema of question
{{
    {concat_schema}}}
    
2. question
{{
    "question": "{question}"
}}

### Let's think step by step.
"""

    return system_prompt, user_prompt


def merge_consecutive_messages(messages):
    if not messages:
        return []

    merged_messages = []
    prev_role = None
    prev_content = ""

    for message in messages:
        current_role = message["role"]
        current_content = message["content"]

        if current_role == prev_role:
            prev_content += "\n" + current_content
        else:
            if prev_role is not None:
                merged_messages.append({"role": prev_role, "content": prev_content.strip()})
            prev_role = current_role
            prev_content = current_content

    if prev_role is not None:
        merged_messages.append({"role": prev_role, "content": prev_content.strip()})

    return merged_messages



def generate_reply(input, model_name, temperature, openai_api_key="", deepseek_api_key=""):

    if debug_print: print(input)
    input = merge_consecutive_messages(input)

    if "gpt" in model_name:
        client = OpenAI(api_key=openai_api_key)
    else:        
        client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model=model_name,
        messages=input,
        temperature=temperature,
        stream=False,
        max_tokens=8192 if model_name == 'deepseek-reasoner' else None
    )

    response = response.choices[0].message.content

    if debug_print: print(response)
    return response


def generate_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' order by tbl_name;")
    schemas = cursor.fetchall()
    schema = ""
    for sc in schemas:
        schema += (sc[0] + "\n\n")
    return schema


def read_schema_description(csv_path, db_path, num_of_sampling=30):
    schema_description = ""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    files = sorted(os.listdir(csv_path))
    for csv_file in files:
        file_path = os.path.join(csv_path, csv_file)
        table_name = os.path.splitext(csv_file)[0]  

        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = detect(raw_data)['encoding']
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = [line.replace('\x00', '').replace('\n', ' ').replace('\r', ' ').strip() for line in f if line.strip()]

                csv_reader = csv.reader(lines)
                headers = next(csv_reader)

                for row in csv_reader:
                    if row:
                        row_description = ", ".join([f"{header}: {value}" for header, value in zip(headers, row)])
                        row_description = "   ### " + row_description
                        schema_description += row_description

                        try:
                            sql = f"select lower(type) from pragma_table_info('{table_name}') where name='{row[0].strip()}';"
                            cursor.execute(sql)
                            column_values = cursor.fetchall()
                            if "blob" in column_values:
                                schema_description += '\n'
                            else:
                                sql = f"SELECT distinct case when substr(`{row[0].strip()}`,1,100) is null then 'null' else substr(`{row[0].strip()}`,1,100) end FROM (SELECT `{row[0].strip()}` FROM `{table_name}` limit 1000) limit {num_of_sampling};"
                                cursor.execute(sql)
                                column_values = cursor.fetchall()
                                column_value = ""
                                for value in column_values:
                                    column_value += (str(value[0]).replace('\n', ' ') + ", ")
                                schema_description = schema_description + "   ### column value examples: " + column_value + '\n'
                        except Exception as e:
                            schema_description += '\n'

        except Exception as e:
            print(f"Warning) Error reading {csv_file}: {e}")
        
    
    conn.close()
    return schema_description



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
                json_pattern = r"```(?:json)?\s*(.*?)\s*```"
                matches = re.findall(json_pattern, response_content, re.DOTALL)
                try:
                    response_json = json.loads(matches[-1])
                    return response_json.get(item, f"Warning) No {item} found")
                except:
                    return "Warning) Response is not in JSON format"


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


def can_be_converted(x):
    try:
        float(x)
        return 'numeric'
    except:
        pass

    date_formats = ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%d-%m-%Y', '%d/%m/%Y', '%Y/%m/%d', '%m/%d/%Y']

    for date_format in date_formats:
        try:
            datetime.strptime(x, date_format)
            return f'date ({date_format})'
        except:
            continue

    return 'neither'


def The_closest_in_edit_distance(word1, word2):
    return jellyfish.jaro_winkler_similarity(str(word1), str(word2))


def get_sample_query_result(schema_value_pair, db_path):

    num_of_sampling = 10
    result = []
    if schema_value_pair != None and isinstance(schema_value_pair, list):

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        conn.create_function("The_closest_in_edit_distance", 2, The_closest_in_edit_distance)

        for pair in schema_value_pair:
            if "." not in str(pair['schema']) or str(pair['schema']).count('.') >= 2: # not table.column form
                continue
            table, column = str(pair['schema']).split('.')
            value = pair['value']
            sqls = []

            sqls.append([f"{num_of_sampling} sample value of `{table}`.`{column}`", f"select distinct substr(`{column}`,1,100) FROM (SELECT `{column}` FROM `{table}` limit 1000) limit {num_of_sampling};"])
            if value != 'null' and value != None and can_be_converted(value) == 'neither':
                sqls.append([f"The value in the `{table}`.`{column}` closest to '{value}' in edit distance", f'select `{column}` from `{table}` ORDER BY The_closest_in_edit_distance(`{column}`, "{value}") DESC LIMIT 1;'])
                sqls.append([f"The results retrieved with the '%{value}%' condition", f"select distinct `{column}` from `{table}` where lower(`{column}`) like '%{str(value).lower().replace(' ','%')}%' limit {num_of_sampling};"])
                                    
            for sql_desc, sql in sqls:
                try:
                    cursor.execute(sql)
                    sql_results = cursor.fetchall()
                    
                    result.append({'schema':pair['schema'], 'value':pair['value'], 'sql_desc':sql_desc, 'sample_sql':sql, 'sql_results':sql_results})
                except sqlite3.Error as e:
                    result.append({'schema':pair['schema'], 'value':pair['value'], 'sql_desc':sql_desc, 'sample_sql':sql, 'sql_results':e})

        conn.close()

    return result





class SimilarQuestionFinder:
    def __init__(self, train_json_all, model):
        embedding_model_name = model
        self.train_data = [
            item for item in train_json_all 
            if item["evidence"].lower() not in ["", "false;"] 
            and item["db_id"].lower() not in [
                "book_publishing_company", "books", "hockey", "movie_3", "movie_4", 
                "public_review_platform", "soccer_2016", "works_cycles"
            ]
        ]
        self.questions = [item["question"] for item in self.train_data]
        self.db_ids = [item["db_id"] for item in self.train_data]
        self.evidences = [item["evidence"] for item in self.train_data]
        self.model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
        self.embeddings = self.model.encode(self.questions, convert_to_tensor=True)
        
    def find_similar_questions(self, target_question, top_k=5, top_n=5):
        with torch.no_grad():
            target_embedding = self.model.encode([target_question], convert_to_tensor=True)
            similarities = self.model.similarity_pairwise(target_embedding, self.embeddings).squeeze(0)
            sorted_indices = torch.argsort(similarities, descending=True).tolist()

        unique_db_ids, top_k_questions = set(), []
        for idx in sorted_indices:
            db_id = self.db_ids[idx]
            if db_id not in unique_db_ids:
                unique_db_ids.add(db_id)
                top_k_questions.append((self.questions[idx], self.db_ids[idx], self.evidences[idx]))
                if len(top_k_questions) == top_k:
                    break
        
        top_k_n_questions = []
        for question, db_id, evidence in top_k_questions:
            same_db_indices = [i for i, db in enumerate(self.db_ids) if db == db_id]
            same_db_similarities = similarities[same_db_indices]
            additional_indices = torch.topk(same_db_similarities, k=top_n, largest=True).indices.tolist()[1:]
            additional_questions = [(self.questions[same_db_indices[idx]], self.evidences[same_db_indices[idx]]) for idx in additional_indices]
            top_k_n_questions.append((question, db_id, evidence, additional_questions))
            
        return top_k_n_questions
    



def evidence_generation(question, concat_schema, finder, data, openai_api_key, deepseek_api_key):

    if schema_summary:
        summary_system_prompt, summary_user_prompt = make_schema_summary_prompt(question, concat_schema)
        summary_prompt = [{"role": "system", "content": summary_system_prompt}, {"role": "user", "content": summary_user_prompt}]

        response = None
        while response is None:
            try:
                response = generate_reply(summary_prompt, schema_summary_llm, schema_summary_temperature, openai_api_key, deepseek_api_key)
            except Exception as e:
                print('api error, wait for 3 seconds and retry...')
                print(e)
                time.sleep(3)
                pass
            
        concat_schema = extract_from_json(response, "summarized_schema")

    evidence_generation_system_prompt, evidence_generation_user_prompt, keyword_extract_system_prompt, keyword_extract_user_prompt = make_prompt(
        question, concat_schema
    )

    keyword_extract_prompt = [{"role": "system", "content": keyword_extract_system_prompt}]
    keyword_extract_prompt.append({"role": "user", "content": keyword_extract_user_prompt})

    evidence_generation_prompt = [{"role": "system", "content": evidence_generation_system_prompt}]

    similar_questions = finder.find_similar_questions(data['question'], opt.top_k)
    sample_num = 0
    for question, db_id, evidence, additional_questions in similar_questions:
        train_schema = generate_schema(f"{opt.train_db_path}/{db_id}/{db_id}.sqlite")
        train_schema_description = read_schema_description(f"{opt.train_db_path}/{db_id}/database_description", f"{opt.train_db_path}/{db_id}/{db_id}.sqlite", 3)

        questions = f"""
"question_0": "{question}"
"""
        for i, (additional_question, _) in enumerate(additional_questions):
            questions += f"""
"question_{i+1}": "{additional_question}"
"""

        concat_train_schema = concat_schema_and_desc(train_schema, train_schema_description)

        if schema_summary:
            summary_system_prompt, summary_user_prompt = make_schema_summary_prompt(questions, concat_train_schema)
            summary_prompt = [{"role": "system", "content": summary_system_prompt}, {"role": "user", "content": summary_user_prompt}]

            response = None
            while response is None:
                try:
                    response = generate_reply(summary_prompt, schema_summary_llm, schema_summary_temperature, openai_api_key, deepseek_api_key)
                except Exception as e:
                    print('api error, wait for 3 seconds and retry...')
                    print(e)
                    time.sleep(3)
                    pass
                
            concat_train_schema = extract_from_json(response, "summarized_schema")

        sample_num += 1
        train_sample_user = f"""
### few-shot sample {sample_num} ####################################################
1. DB Schema of Samples
{{
{concat_train_schema}
}}

2. Question and evidence pair samples
{{
"question": "{question}",
"evidence": "{evidence}"
}}

"""

        for (additional_question, additional_evidence) in additional_questions:
                train_sample_user += f"""
{{
"question": "{additional_question}",
"evidence": "{additional_evidence}"
}}

"""

        train_sample_user += f"""
##################################################################
"""
        evidence_generation_prompt.append({"role": "user", "content": train_sample_user})
        
    response = None
    while response is None:
        try:
            response = generate_reply(keyword_extract_prompt, keyword_extract_llm, keyword_extract_temperature, openai_api_key, deepseek_api_key)
        except Exception as e:
            print('api error, wait for 3 seconds and retry...')
            print(e)
            time.sleep(3)
            pass

    schema_value_pair = extract_from_json(response, "schema-value-pair")

    sample_query_result = get_sample_query_result(schema_value_pair, f"{opt.db_path}/{data['db_id']}/{data['db_id']}.sqlite")

    sample_sql_result_prompt = """
### Sample SQL results for Data Exploration ####################################################
"""
    for item in sample_query_result:
        sample_sql_result_prompt += f"# {item['sql_desc']}: {item['sql_results']}\n"
    sample_sql_result_prompt += """##################################################################

"""

    evidence_generation_prompt.append({"role": "user", "content": sample_sql_result_prompt})
    evidence_generation_prompt.append({"role": "user", "content": evidence_generation_user_prompt})

    response = None
    while response is None:
        try:
            response = generate_reply(evidence_generation_prompt, evidence_generation_llm, evidence_generation_temperature, openai_api_key, deepseek_api_key)
        except Exception as e:
            print('api error, wait for 3 seconds and retry...')
            print(e)
            time.sleep(3)
            if str(e).startswith("This model's maximum context length is") and len(evidence_generation_prompt) > 2:
                evidence_generation_prompt.pop(1)
            pass
    evidence = str(extract_from_json(response, "evidence")).replace('\n',', ')

    if isinstance(evidence, dict):
        evidence = ", ".join([f"{key}: {value}" for key, value in evidence.items()])
    return evidence


def process_data(i, data, opt, finder, openai_api_key, deepseek_api_key):
    schema = generate_schema(f"{opt.db_path}/{data['db_id']}/{data['db_id']}.sqlite")
    schema_description = read_schema_description(
        f"{opt.db_path}/{data['db_id']}/database_description",
        f"{opt.db_path}/{data['db_id']}/{data['db_id']}.sqlite",
        3
    )
    concat_schema = concat_schema_and_desc(schema, schema_description)

    evidence = evidence_generation(data['question'], concat_schema, finder, data, openai_api_key, deepseek_api_key)

    data["evidence"] = evidence
    print("===================================================================================================")
    print(f"idx: {i}")
    print(f"question: {data['question']}")
    print(f"evidence: {data['evidence']}")
    print("===================================================================================================")
    sys.stdout.flush()

    return i, data



if __name__ == "__main__":
    opt = parse_option()
    print(opt)

    with open(opt.dataset_json_path, 'rb') as f:
        raw_data = f.read()
        encoding = detect(raw_data)['encoding']

    with open(opt.dataset_json_path, encoding=encoding) as f:
        question_json_all = json.load(f)
    with open(opt.train_json_path, encoding=encoding) as f:
        train_json_all = json.load(f)

    finder = SimilarQuestionFinder(train_json_all, embedding_model_name)

    res = [None] * len(question_json_all) 
    completed_count = 0

    with ThreadPoolExecutor(max_workers=parallel_process) as executor:
        future_to_index = {executor.submit(process_data, i, data, opt, finder, opt.openai_api_key, opt.deepseek_api_key): i for i, data in enumerate(question_json_all)}

        with tqdm(total=len(question_json_all), desc=f"Done: {completed_count}/{len(question_json_all)}") as pbar:
            for future in as_completed(future_to_index):
                try:
                    i, processed_data = future.result()
                    res[i] = processed_data
                    
                    completed_count += 1
                    pbar.set_description(f"Done: {completed_count}/{len(question_json_all)}")
                    pbar.update(1)
                except Exception as e:
                    print(f"[ERROR] Subprocess failed: {str(e)}", file=sys.stderr)
                    with open(opt.output_path, 'w', encoding=encoding) as f:
                        json.dump(res, f, indent=2)
                    for f in future_to_index:
                        f.cancel()
                    sys.exit(1)

    with open(opt.output_path, 'w', encoding=encoding) as f:
        json.dump(res, f, indent=2)
