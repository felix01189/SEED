import json
from openai import OpenAI
import time
from charset_normalizer import detect
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

###settting#####################################################################################
temperature = 0.
llm_model = "deepseek-chat"
input_path = "./bird_dev_seed_deepseek_evi.json"
output_path = "./bird_dev_seed_deepseek_revised_evi.json"
deepseek_api_key = ""
################################################################################################

def make_prompt(question, evidence):
    system_prompt = """### As a data science expert, you have generated evidence to support text-to-SQL tasks.
However, it has been observed that the evidence related to "join" hinders SQL generation rather than assisting it.
Therefore, check the content of the evidence for the given question.
If the evidence does not contain any references to "join," output the evidence as is.
If it does, remove the parts related to "join" and generate the revised evidence.
Output the result as a string only, without any explanation.
"""

    user_prompt = f"""### input:
{{
"question": {question},
"evidence": {evidence}
}}

### revised evidence: 
"""
    return system_prompt, user_prompt

def generate_reply(input, model_name, temperature, deepseek_api_key=""):
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model=model_name,
        messages=input,
        temperature=temperature,
        stream=False,
        max_tokens=1000
    )

    response = response.choices[0].message.content

    print(response)

    return response


def evidence_shortening(question, evidence, deepseek_api_key):
    system_prompt, user_prompt = make_prompt(question, evidence)

    shortening_prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    response = None
    while response is None:
        try:
            response = generate_reply(shortening_prompt, llm_model, temperature, deepseek_api_key)
        except Exception as e:
            print('API error, wait for 3 seconds and retry...')
            print(e)
            time.sleep(3)
            pass

    if "### evidence:\n" in response:
        revised_evidence = response.split("### evidence:\n")[1]
    elif "### evidence: " in response:
        revised_evidence = response.split("### evidence: ")[1]
    else:
        revised_evidence = response

    if isinstance(revised_evidence, dict):
        revised_evidence = ", ".join([f"{key}: {value}" for key, value in revised_evidence.items()])

    print("===================================================================================================")
    print(f"question: {question}")
    print(f"evidence: {evidence}")
    print(f"revised_evidence: {revised_evidence}")
    print(f"length diff) original: {len(evidence)}, revised: {len(revised_evidence)}")
    print("===================================================================================================")

    return revised_evidence



if __name__ == "__main__":

    with open(input_path, 'rb') as f:
        raw_data = f.read()
        encoding = detect(raw_data)['encoding']
    with open(input_path, encoding=encoding) as f:
        input_json = json.load(f)

    res = []
    changed_count = 0
    total_tasks = len(input_json)

    print(f"Starting evidence shortening for {total_tasks} tasks...")

    with tqdm(total=total_tasks, desc="Processing tasks", unit="task") as pbar:
        for idx, data in enumerate(input_json, start=1):
            revised_evidence = evidence_shortening(data['question'], data['evidence'], deepseek_api_key)

            if len(data['evidence']) != len(revised_evidence):
                changed_count += 1

            data['evidence'] = revised_evidence
            res.append(data)
            pbar.update(1)

    print("All tasks completed. Saving output...")

    with open(output_path, 'w', encoding=encoding) as f:
        json.dump(res, f, indent=2)

    print(f"Output saved to {output_path}")

    print("===================================================================================================")
    print(f"Total tasks: {total_tasks}")
    print(f"Cases where evidence length changed: {changed_count}")
    print("===================================================================================================")
