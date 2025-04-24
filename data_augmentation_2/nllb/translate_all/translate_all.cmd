transfer_executable = false
should_transfer_files = NO
initialdir        = /home2/dinuoz/Ling573
# translate.sub
executable       = translate_all.sh
getenv     = true
output           = translate_rn.out
error            = translate_rn.err
log              = translate_rn.log


# resources
request_cpus     = 4
request_memory   = 8GB
request_gpus     = 1

# pass in:  $1=input  $2=output  $3=seed
arguments        = toy_data_tokenized.jsonl translate_all_dataset.jsonl --use_gpu

queue
