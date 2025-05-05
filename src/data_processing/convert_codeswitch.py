import json

def convert_codeswitch_to_sample(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            obj = json.loads(line)
            input_text = obj.get("Text", "").strip()
            sample = {
                "input_text": input_text,
                "summary_text": ""
            }
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_path = "src/stress_test/codeswitch_test_100.jsonl"
    output_path = "src/stress_test/codeswitch_sample_100.jsonl"
    convert_codeswitch_to_sample(input_path, output_path)
    print(f"Converted file saved to {output_path}")
