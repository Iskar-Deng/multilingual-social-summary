transfer_executable = false
should_transfer_files = NO
# translate.sub
executable       = translate_full.sh
getenv     = true
output           = translate.out
error            = translate.err
log              = translate.log


# resources
request_cpus     = 4
request_memory   = 8GB
request_gpus     = 1

# pass in:  $1=input  $2=output  $3=seed
arguments        = toy_data_tokenized.jsonl translate_full.jsonl --use_gpu



queue
