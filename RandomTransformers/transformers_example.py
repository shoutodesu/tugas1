# transformers_example.py

from transformers import pipeline, AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
import requests
from PIL import Image
import torch

def main():
    # Text Classification
    print("Text Classification:")
    classifier = pipeline('sentiment-analysis')
    result = classifier('I love using the transformers library!')
    print(result)

    # Text Generation with GPT-2
    print("\nText Generation with GPT-2:")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    text = "Once upon a time"
    inputs = gpt2_tokenizer(text, return_tensors="pt")
    outputs = gpt2_model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)
    generated_text = gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(generated_text)

    # Object Detection
    print("\nObject Detection:")
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"
    image = Image.open(requests.get(url, stream=True).raw)
    
    object_detector = pipeline('object-detection')
    results = object_detector(image)
    for result in results:
        print(f"Detected {result['label']} with confidence {result['score']:.2f} at location {result['box']}")

if __name__ == "__main__":
    main()
