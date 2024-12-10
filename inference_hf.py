from openai import AsyncOpenAI
from tqdm import tqdm
import random
import base64
import argparse
import re
import os
from prompt import prompt
import asyncio
from datetime import datetime
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_random_exponential
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
import numpy as np

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
async def vllm_openai(image_bytes: str, prompt: str, filter_model_name: str, temperature=0.2, top_p=0.2):
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    client = AsyncOpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }
    ]
    try:
        response = await client.chat.completions.create(
            model=filter_model_name,
            messages=messages,
            # timeout=50,
            temperature=temperature,
            top_p=top_p,
            stream=False,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    finally:
        await client.close()

async def process_item(item: Dict, filter_model_name: str, semaphore: asyncio.Semaphore):
    
    random_number = random.choice([0, 1])
    question = prompt(item, random_number)
    question=question.replace("<image>\n","")

    async with semaphore:
        response = await vllm_openai(item["image"]['bytes'], question, filter_model_name)

    rejected_response_index = 1 if random_number == 0 else 0
    rejected_response_choice = np.where(item["human_ranking"] == rejected_response_index )[0][0] + 1

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
    if "meta" not in item:
        item["meta"] = {}
    item["meta"]["filtering model"] = filter_model_name
    item["meta"]["filter_choice"] = response
    item["meta"]["filter_prompt"] = question
    item["meta"]["filter_number"] = flag_choice
    item["meta"]["random_number"] = random_number
    item["meta"]["flag_status"] = flag_status
    result = {
        "id": item["id"],
        "query": item["query"],
        "response": item["response"],
        "ranking": item["human_ranking"],
        "models": item["models"],
        "judge": item["judge"],
        "rationale": item["rationale"],
        "meta": item["meta"],
        "query_source":item["query_source"],
    }

    return result

async def parallel_process(
    df: pd.DataFrame,
    filter_model_name: str,
    output_path: str,
    k: int,
    num_processes: int
):
    semaphore = asyncio.Semaphore(num_processes)
    all_results = []
    tasks = [process_item(item=row.to_dict(),filter_model_name=filter_model_name,semaphore=semaphore) for _, row in df.iterrows() for _ in range(k)]
    all_results = await tqdm_asyncio.gather(*tasks, desc="Processing items")
    results_df = pd.DataFrame(all_results)
    results_df.to_json(output_path, orient='records', lines=True)

async def main():
    parser = argparse.ArgumentParser(description="Process some input parameters.")
    parser.add_argument('--data_path', type=str, default="test-00000-of-00001.parquet", help='Path to the input data file')
    parser.add_argument('--filter_model_name', type=str, default="Qwen/Qwen2-VL-7B-Instruct", help='Name of the filter model')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--num_processes', type=int, default=90)
    args = parser.parse_args()

    data_path = args.data_path
    filter_model_name = args.filter_model_name
    output_path = f"{filter_model_name}-{datetime.now().strftime('%m%d')}.jsonl"
    output_path = output_path.replace("/","-")
    print(output_path)
    k = args.k
    num_processes = args.num_processes
    df = pd.read_parquet(data_path)
    if 'meta' in df.columns:
        df = df.drop('meta', axis=1)
    await parallel_process(
        df,filter_model_name, output_path, k, num_processes
    )

if __name__ == "__main__":
    asyncio.run(main())