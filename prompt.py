def prompt(data_obj,random_number):
    answers = [data_obj["response"][0], data_obj["response"][1]] if random_number == 0 else [data_obj["response"][1], data_obj["response"][0]]
    prompt_str = f''' You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions. Please analyze the following image and question, then determine which of the two provided answers is better.

Question: {data_obj["query"]}

Answer 1: {answers[0]}

Answer 2: {answers[1]}

Please evaluate both answers based on the following criteria:
1. Accuracy: How well does the answer align with the visual information in the image?
2. Completeness: Does the answer fully address all aspects of the question?
3. Clarity: Is the answer easy to understand and well-articulated?
4. Relevance: Does the answer directly relate to the question and the image?

After your evaluation, please:
1. Explain your reasoning for each criterion.
2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: Overall Judgment: Answer X is better.

Your response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task.'''
    return prompt_str
    