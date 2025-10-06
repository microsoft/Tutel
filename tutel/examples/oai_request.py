#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import openai
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prompt', type=str, default='Hello!')
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--api_key', type=str, default='')
parser.add_argument('--url', type=str, default='0.0.0.0:8000')
args = parser.parse_args()

client = openai.OpenAI(base_url=f'http://{args.url}/v1', api_key=args.api_key)

response = client.chat.completions.create(
  model=args.model,
  messages=[{"role": "user", "content": args.prompt}],
)

print('Prompt:', args.prompt + '\n')
print('Response:\n', response.choices[0].message.content)
