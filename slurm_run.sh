#!/bin/bash

cd /home/lz307/rds/hpc-work/
singularity exec --bind $(pwd):/GenieRedux --bind ~/.cache:/.cache --nv --env WANDB_API_KEY=3f8710e1350e8de5755e84bee294a66ab17b765f --writable-tmpfs geanie_redux.sif bash -c "cd GenieRedux && ./run.sh --config=tokenizer.yaml --num_processes=4 --train.batch_size=12 --train.grad_accum=2 --train.num_train_steps=50000"
