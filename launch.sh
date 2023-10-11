#!/bin/bash

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python ${script_dir}/train.py >> /out/accuracy.out
