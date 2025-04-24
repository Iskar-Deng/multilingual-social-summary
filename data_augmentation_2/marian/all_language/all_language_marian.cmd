transfer_executable = false
should_transfer_files = NO
# translate.sub
executable       = run_all_language_marian.sh
getenv     = true
output           = translate.out
error            = translate.err
log              = translate.log


# resources
request_cpus     = 4
request_memory   = 8GB
request_gpus     = 1

# pass in:  $1=input  $2=output  $3=seed
arguments        = toy_data_tokenized.jsonl all_translated_dataset.jsonl  --use_gpu

queue
