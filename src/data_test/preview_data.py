import json

FILE_PATH = "/gscratch/stf/mx727/corpus-webis-tldr-17.json"
NUM_LINES = 5

with open(FILE_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= NUM_LINES:
            break
        try:
            data = json.loads(line)
            print(f"--- Line {i + 1} ---")
            for k, v in data.items():
                v_str = str(v)
                preview = v_str[:200] + ("..." if len(v_str) > 200 else "")
                print(f"{k}: {preview}")
            print()
        except json.JSONDecodeError:
            print(f"[Warning] Line {i + 1} is not valid JSON.")
