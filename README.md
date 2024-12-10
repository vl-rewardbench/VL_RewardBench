### data
please download data from MMInstruction/VL-RewardBench (https://huggingface.co/datasets/MMInstruction/VL-RewardBench/)

# inference
We recommend using vLLM as the inference engine.(https://github.com/vllm-project/vllm) 
Once you have deployed your model using vLLM (make sure to set the port to 8000), you can use inference_hf.py to perform inference on our dataset.
```
python inference_hf.py \
    --data_path "test-00000-of-00001.parquet" \
    --filter_model_name "Qwen/Qwen2-VL-7B-Instruct" \
    --k 10
```

# Metric
You can use `cal.py` to evaluate the model's performance. Simply update the data path in the script to match your dataset.