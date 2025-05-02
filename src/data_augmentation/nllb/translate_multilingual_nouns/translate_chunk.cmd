# translate_chunks.sub

transfer_executable = false
should_transfer_files = NO

executable = translate_muiltilingual_nouns.sh
getenv = true

output = logs/job_$(Cluster).$(Process).out
error  = logs/job_$(Cluster).$(Process).err
log    = logs/job_$(Cluster).$(Process).log

request_cpus = 4
request_memory = 16GB
request_gpus = 1

+GPUJob = true
+IsCUDAJob = true


arguments = $(input) $(outfile) 123 --use_gpu

queue input, outfile from file_list.txt
