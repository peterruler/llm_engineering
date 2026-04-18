#!/bin/bash
lms server start --port 1234
# either one or the other following:
lms load google/gemma-4-31b --context-length 32768
lms load Qwen/Qwen3.5-35B-A3B --context-length 32768
lms load openai/gpt-oss-120b --context-length 32768
