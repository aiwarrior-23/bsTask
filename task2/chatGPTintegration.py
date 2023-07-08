from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS
import openai

app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
chat_history_ids = None
openai.api_key = "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"


@app.route('/chat', methods=['POST'])
def chat():
    global chat_history_ids
    user_input = request.json['user_input']
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({'response': response})

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data['prompt']
    n = data['n']
    size = data['size']

    response = openai.Image.create(
        prompt=prompt,
        n=n,
        size=size
    )

    return jsonify(response)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
