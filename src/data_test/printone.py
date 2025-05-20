import json

with open("/gscratch/stf/mx727/tldr_head_10000.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i == 0:  # change to another number to see a different line
            example = json.loads(line)
            print("🔍 Raw Example:")
            for k, v in example.items():
                print(f"{k}: {str(v)[:200]}{'...' if len(str(v)) > 200 else ''}")
            break
