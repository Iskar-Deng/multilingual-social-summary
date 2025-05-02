transfer_executable = false
should_transfer_files = NO
# translate.sub
executable       = translate_multilingual_nouns.sh
getenv     = true
output           = translate.out
error            = translate.err
log              = translate.log


# resources
request_cpus     = 4
request_memory   = 36GB
request_gpus     = 1
+GPUJob = true
+IsCUDAJob = true

# pass in:  $1=input  $2=output  $3=seed
arguments        = filtered_jsonl_splits output_folder 123 --use_gpu

queue

