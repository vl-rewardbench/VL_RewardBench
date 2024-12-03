import json
import random
import time
from openai import OpenAI
import os 
from tqdm import tqdm 

import re 
PARSE_PROMPT = """
You are given a pairwise judgement for two responses. Please return the better response according to the judgement.
Return the Answer X ONLY. e.g., Answer 1 or Answer 2.

Judgement: {judgement}
"""
openai_api_key = os.environ.get("OPENAI_API_KEY", "YOUR_KEY") 
client = OpenAI(
            api_key=openai_api_key,
            # base_url=openai_api_base,
        )
def parse_by_llm(response, model="gpt-4o-mini"):
    # get the judgement from response using gpt-4o
    creator = client.chat.completions
    messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PARSE_PROMPT.format(judgement=response)},
                    
                ],
            }
        ]
    response = creator.create(
        model=model,
        messages=messages,
        max_tokens=32)
    return response.choices[0].message.content.strip() 



def shuffle_data(data):
    # random.seed(time.time())
    for i in range(len(data)):
        j = random.randint(0, len(data) - 1)
        data[i], data[j] = data[j], data[i]
    return data

def distribution(input_path, k=10, reparse=False, use_llm_parse=False):
    distribution = {}
    data = []
    try:
        with open(input_path, "r") as f:
            data = [json.loads(line) for line in f]
            if "0" in data[0]:
                data = [item["0"] for item in data]
            
    except FileNotFoundError:
        print(f"File not found: {input_path}")
        return distribution
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return distribution
    
    data = shuffle_data(data)
    
    for item in tqdm(data):
        if item is None:
            continue 
        id = item["id"] + item["query"] + "\t".join(sorted(item["response"]))
        dataset = item["image_path"].split("/")[1]
        flag_status = item["meta"]["flag_status"]
        # 
        if reparse:
            ranking = item.get("ranking", [] )
            assert len(ranking) != 0, "ranking is"
            random_number = item["meta"]["random_number"]
            rejected_response_choice = ranking.index(1 if random_number == 0 else 0) + 1 
            response = item["meta"]["filter_choice"] 
            pattern = r"(?:Overall Judgment|Therefore)\s*.*\s*-*\s*Answer\s*(\d+)\s*is\s*(?:the\s*)?(?:slightly\s*)?better"
            match = re.search(pattern, response.replace("\n","").replace("*",""), re.IGNORECASE)
            flag_choice = -1
            flag_status = "doesntMatch"

            if match:
                answer_number = int(match.group(1))
                flag_choice = answer_number
                if flag_choice == rejected_response_choice:
                    flag_status = "reject"
                else:
                    flag_status = "agree"

        if id not in distribution:
            if flag_status == "agree":
                distribution[id] = {"agree": 1, "reject": 0, "doesntMatch": 0, "dataset": dataset}
            elif flag_status == "reject":
                distribution[id] = {"agree": 0, "reject": 1, "doesntMatch": 0, "dataset": dataset}
            elif flag_status == "doesntMatch":
                distribution[id] = {"agree": 0, "reject": 0, "doesntMatch": 1, "dataset": dataset}
                if use_llm_parse:
                    parsed_response = parse_by_llm(item["meta"]["filter_choice"])
                    random_number = item['meta']['random_number']
                    rejected_response_choice = item["ranking"].index(1 if random_number == 0 else 0) + 1

                    if "Answer 1".lower() in parsed_response.lower():
                        judge_ret = 1 
                    elif "Answer 2".lower() in parsed_response.lower():
                        judge_ret = 2 
                    else:
                        print("Error in parsing response", parsed_response)
                        distribution[id] = {"agree": 0, "reject": 0, "doesntMatch": 1, "dataset": dataset}
                        continue 
                    #     # continue 
                    if judge_ret == rejected_response_choice: 
                        distribution[id] = {"agree": 0, "reject": 1, "doesntMatch": 0, "dataset": dataset}
                    else:
                        distribution[id] = {"agree": 1, "reject": 0, "doesntMatch": 0, "dataset": dataset}


        else:
            if distribution[id]["agree"] + distribution[id]["reject"] + distribution[id]["doesntMatch"] >= k:
                continue
            if flag_status == "agree":
                distribution[id]["agree"] += 1
            elif flag_status == "reject":
                distribution[id]["reject"] += 1
            elif flag_status == "doesntMatch":
                distribution[id]["doesntMatch"] += 1
                if use_llm_parse: #
                    parsed_response = parse_by_llm(item["meta"]["filter_choice"])
                    random_number = item['meta']['random_number']
                    rejected_response_choice = item["ranking"].index(1 if random_number == 0 else 0) + 1

                    if "Answer 1".lower() in parsed_response.lower():
                        judge_ret = 1 
                    elif "Answer 2".lower() in parsed_response.lower():
                        judge_ret = 2 
                    else:
                        print("Error in parsing response", parsed_response)
                        distribution[id]['doesntMatch'] += 1 
                        continue 
                        # continue 
                    if judge_ret == rejected_response_choice: 
                        distribution[id]['reject'] += 1 
                    else:
                        distribution[id]['agree'] += 1 
            
    return distribution

def filter_out_doesntMatch(distribution):
    filtered_distribution = {}
    for id in distribution:
        if distribution[id]["doesntMatch"] == 0:
            filtered_distribution[id] = distribution[id]
       
            
    return filtered_distribution

def classify(distribution, k=10):
    ban_count = 0
    reject_count = 0
    agree_count = 0
    doesntMatch_count = 0
    other_count = 0

    for id in distribution:
        agree = distribution[id]["agree"]
        reject = distribution[id]["reject"]
        doesntMatch = distribution[id]["doesntMatch"]
        total_count = agree + reject + doesntMatch
        distribution[id]["total_count"] = total_count

        if total_count < k:
            distribution[id]["classify"] = "banned"
            ban_count += 1

        if agree > total_count / 2:
            distribution[id]["classify"] = "agree"
            agree_count += 1
        elif reject > total_count / 2:
            distribution[id]["classify"] = "reject"
            reject_count += 1
        elif doesntMatch > total_count / 2:
            distribution[id]["classify"] = "doesntMatch"
            doesntMatch_count += 1
        else:
            distribution[id]["classify"] = "other"
            other_count += 1
    agree_rate = agree_count / (len(distribution) + 1e-6)
    reject_rate = reject_count / (len(distribution) + 1e-6)
    doesntMatch_rate = doesntMatch_count / (len(distribution) + 1e-6)
    other_rate = other_count / (len(distribution) + 1e-6) 

    normalized_agree_rate = agree_count / (agree_count + reject_count + 1e-6)
    normalized_reject_rate = reject_count / (agree_count + reject_count + 1e-6)
    print(f"k={k} ----------------")
    print(f"Number of total items: {len(distribution)}")
    # print(f"Number of banned items: {ban_count}")
    print(f"Number of agree items: {agree_count}, rate: {round(agree_rate,4)}, normalized rate: {round(normalized_agree_rate,4)}")
    print(f"Number of reject items: {reject_count}, rate: {round(reject_rate,4)}, normalized rate: {round(normalized_reject_rate,4)}")
    print(f"Number of doesntMatch items: {doesntMatch_count}, rate: {round(doesntMatch_rate,4)}")
    return distribution
from collections import defaultdict 

def distribution_dataset(distribution):
    dataset_distribution = {}
    for id in distribution:
        dataset = distribution[id]["dataset"]
        if dataset not in dataset_distribution:
            dataset_distribution[dataset] = {"agree_number": 0, "reject_number": 0, "doesntMatch_number": 0, "other_number": 0, "reject_rate": -1}
        if distribution[id]["classify"] == "agree":
            dataset_distribution[dataset]["agree_number"] += 1
        elif distribution[id]["classify"] == "reject":
            dataset_distribution[dataset]["reject_number"] += 1
        elif distribution[id]["classify"] == "doesntMatch":
            dataset_distribution[dataset]["doesntMatch_number"] += 1
        elif distribution[id]["classify"] == "other":
            dataset_distribution[dataset]["other_number"] += 1
    group_mapping = {
        "vlfeedback": "general",
        "povid": "hallucination",
        "reasoning_tasks": "reasoning",
        "rlhf-v": "hallucination",
        "rlaif-v": "hallucination",
        "wildvision-battle": "general"
    }
    group_correct = defaultdict(lambda :0)
    group_total = defaultdict(lambda : 0)

    for dataset in dataset_distribution:
        dataset_distribution[dataset]["accept_rate"] = round((dataset_distribution[dataset]["agree_number"] / (dataset_distribution[dataset]["agree_number"] + dataset_distribution[dataset]["reject_number"])), 4)
        group_correct[group_mapping[dataset]] += dataset_distribution[dataset]["agree_number"] 
        group_total[group_mapping[dataset]] += dataset_distribution[dataset]["agree_number"] 
        group_total[group_mapping[dataset]] += dataset_distribution[dataset]["reject_number"] 
        
    # DATASET level accuracy 
    # print(
    #     dataset_distribution['rlaif-v']['accept_rate'],
    #     dataset_distribution['rlhf-v']['accept_rate'],
    #     dataset_distribution['povid']['accept_rate'],
    #     dataset_distribution['vlfeedback']['accept_rate'],
    #     dataset_distribution['wildvision-battle']['accept_rate'],
    # )

    # for group in ["general", "hallucination", "reasoning"]:
    print("general\t hallucination\t reasoning")
    print( "{:.4f} {:.4f} {:.4f}".format(group_correct["general"] / group_total["general"],  group_correct["hallucination"] / group_total["hallucination"],group_correct["reasoning"] / (group_total["reasoning"] + 1e-6)))
    return dataset_distribution

def main():
    random.seed(1234)
    input_path =  "./data/infer_results/gpt-4o-1023.jsonl"
    print(input_path)

    for i in [1, 3, 5, 7,9]:
        distributions = distribution(input_path, k=i, reparse=True, use_llm_parse=False)
        filtered_distributions = filter_out_doesntMatch(distributions)
        classified_distributions = classify(distributions, k=i)
        dataset_distribution = distribution_dataset(classified_distributions)

if __name__ == "__main__":
    main()
