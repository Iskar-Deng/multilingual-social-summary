from datasets import load_from_disk

# Load the tokenized dataset directory (NOT the .arrow file itself)
ds = load_from_disk("/gscratch/stf/mx727/multilingual-social-summary/data/splits/tokenized_train")

print(f"✅ Loaded dataset with {len(ds)} examples")
print(f"📂 Original columns: {ds.column_names}")

# Remove unnecessary columns
ds = ds.remove_columns([
    'author', 'body', 'content_len', 'id', 'normalizedBody',
    'subreddit', 'subreddit_id', 'summary_len', 'title'
])

# Save the cleaned dataset to a new path
clean_path = "/gscratch/stf/mx727/multilingual-social-summary/data/splits/tokenized_train_clean"
ds.save_to_disk(clean_path)

print(f"✅ Cleaned dataset saved to: {clean_path}")
