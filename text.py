import json

# Correcting the quotation marks around the file name

data = [json.loads(line) for line in open('data/dev.jsonl').read().splitlines()]

print(data[1]['text'])
