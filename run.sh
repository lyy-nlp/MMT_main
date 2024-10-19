#!/bin/bash


start=5
end=50


for ((i=start; i<=end; i++)); do
    # 执行 average_checkpoint.sh
    echo "Executing average_checkpoint.sh with num-epoch-checkpoints=$i ..."
    python3 scripts/average_checkpoints.py \
        --inputs /home/lyy/Gating2-fairseq-inter2/results \
        --num-epoch-checkpoints $i \
        --output /home/lyy/Gating2-fairseq-inter2/results/model.pt \

    #  num-epoch-checkpoints
    echo "num-epoch-checkpoints=$i" >> generate_results.txt

    #  generate.sh
    echo "Executing generate.sh ..."
    python3 generate.py  /gb/lyy/data-after-preprocess \
        --path /home/lyy/Gating2-fairseq-inter2/results/model.pt \
        --source-lang en --target-lang de \
        --beam 5 \
        --num-workers 12 \
        --batch-size 128 \
        --results-path /home/lyy/Gating2-fairseq-inter2/results \
        --remove-bpe \
        --fp16 \
        --nbest 5 \
    >> generate_results.txt
done

echo "Script execution completed."
