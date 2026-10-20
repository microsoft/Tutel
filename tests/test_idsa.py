#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import asyncio
import requests
import time
from datasets import load_dataset
from math_verify import parse, verify

data2hfargs = {
    "gsm8k": {"path": "openai/gsm8k", "name": "main", "split": "test"},
    "gsm8k-platinum": {"path": "madrylab/gsm8k-platinum", "name": "main", "split": "test"},
    "math-500": {"path": "HuggingFaceH4/MATH-500", "split": "test"},
    "aime-2024": {"path": "HuggingFaceH4/aime_2024", "split": "train"},
    "aime-2025": {"path": "math-ai/aime25", "split": "test"},
    "aime-2026": {"path": "math-ai/aime26", "split": "test"},
}

def evaluate(prompt, answer, rank):
  import subprocess, tempfile, os
  env = os.environ.copy()
  env["IS_SANDBOX"] = "1"
  with tempfile.TemporaryDirectory() as workspace:
    with open(f'{workspace}/Q.txt', 'w') as fp:
      fp.write(prompt)
    while True:
      result = subprocess.run(["claude", "--dangerously-skip-permissions", "-p", "Please read the question from ./Q.txt and calculate its answer as required."], cwd=workspace, env=env, capture_output=True, text=True).stdout
      if 'for a proxy or gateway intercepting the request' not in result and 'timed out' not in result:
        break
  result = result.replace('$$', '$')
  return result, answer, prompt, rank

def evaluate_http(prompt, answer, rank):
  text = []
  with requests.post(os.environ.get('ANTHROPIC_BASE_URL', 'http://127.0.0.1:8000') + '/chat', json={"text": prompt}, headers={"x-api-key": "sk-ant-api00-local-mock-key"}, stream=True) as response:
    response.raise_for_status()
    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
      if chunk:
        text.append(chunk)
  result = ''.join(text).replace('$$', '$')
  return result, answer, prompt, rank


if int(os.environ.get('TOOL', 0)) == 0:
  evaluate = evaluate_http

is_sync = int(os.environ.get('SYNC', 0))

def judge(result, answer, Q, rank):
  is_match = False
  try:
    predict = parse(result)[0]
    truth = parse(answer)[0]
    is_match = is_match or (str(truth) in str(predict))
    try:
      if is_match:
        raise
      predict = result.split('$')[-2]
      predict = predict.split('=')[-1]
      if len(predict) > 50:
        raise
      is_match = is_match or (str(truth) in str(predict))
    except:
      pass
    print(f'[{rank}] {predict} v.s. {truth}, is_match = {is_match}', flush=True)
    if not is_match:
      print('========================================================')
      print(f' >> Question (Excepting {answer}):', Q.replace('\n', ' '))
      print('========================================================')
      print(f' >> Wrong Answer:', '.. ' + result[-150:].replace('\n', ' '))
      print('========================================================')
  except Exception as ex:
    print('[Error]', ex, f'<{result}>')
  return is_match

async def main():
  # test = load_dataset(**data2hfargs['aime-2026'])
  DS = os.environ.get('DS', 'aime-2026')
  print('Dataset =', DS)
  test = load_dataset(**data2hfargs[DS])
  passed, total = 0, 0
  tasks = []

  t_start = time.perf_counter()
  for rank, doc in enumerate(test):
    Q = doc.get('problem', doc.get('question', '')) + '\nYou are allowed to use tools to help calculation, but please ensure the time limit per try is at most 1 minute. Finally, please put the final answer (number only) within $$'
    A = str(doc['answer'])
    # if A not in ('60', '240'): continue
    # if rank <= 15: continue
    if is_sync:
      tasks += [await asyncio.to_thread(evaluate, prompt=Q, answer=A, rank=rank)]
      total += 1
      if judge(*tasks[-1]):
        passed += 1
      print(f'>> Progress: {passed} / {total}')
    else:
      tasks += [asyncio.to_thread(evaluate, prompt=Q, answer=A, rank=rank)]

  if not is_sync:
    for coro in asyncio.as_completed(tasks):
      total += 1
      if judge(*await coro):
        passed += 1
      print(f'>> Progress: {passed} / {total}')
  t_stop = time.perf_counter()
  print(f'Time: {t_stop - t_start} sec. Correct: {passed} / {len(test)}.', flush=True)
  return passed

asyncio.run(main())
