import os
import openai
openai.api_key = "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"
print(openai.Image.create(
  prompt="A cute baby sea otter",
  n=2,
  size="1024x1024"
))