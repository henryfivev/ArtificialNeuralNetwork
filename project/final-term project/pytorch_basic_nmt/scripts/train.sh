#!/bin/sh
# 改变分词方式
# 数据增强
# python nmt.py train --cuda --vocab=./data/vocab.json --train-src=./data/train_ch.txt --train-tgt=./data/train_en.txt --dev-src=./data/val_ch.txt --dev-tgt=./data/val_en.txt --input-feed --valid-niter=5500 --label-smoothing=0.1 --dropout=0.5  
# python nmt.py train --cuda --vocab=./data/1/vocab.json --train-src=./data/1/train_ch.txt --train-tgt=./data/1/train_en.txt --dev-src=./data/1/val_ch.txt --dev-tgt=./data/1/val_en.txt --input-feed --valid-niter=100 --label-smoothing=0.1 --dropout=0.5
# 10.7
# python nmt.py train --cuda --vocab=./data/1/vocab.json --train-src=./data/1/train_ch.txt --train-tgt=./data/1/train_en.txt --dev-src=./data/1/val_ch.txt --dev-tgt=./data/1/val_en.txt --input-feed --valid-niter=100 --label-smoothing=0.1 --dropout=0.5 --seed=3407 --hidden-size=512 --embed-size=512
# 10.5
# python nmt.py train --cuda --vocab=./data/1/vocab.json --train-src=./data/1/train_ch.txt --train-tgt=./data/1/train_en.txt --dev-src=./data/1/val_ch.txt --dev-tgt=./data/1/val_en.txt --input-feed --valid-niter=100 --label-smoothing=0.1 --dropout=0.5 --seed=3407 --hidden-size=512 --embed-size=512 --batch-size=16


vocab="data/vocab.json"
train_src="data/train_ch.txt"
train_tgt="data/train_en.txt"
dev_src="data/val_ch.txt"
dev_tgt="data/val_en.txt"
test_src="data/test_ch.txt"
test_tgt="data/test_en.txt"

work_dir="work_dir"

mkdir -p ${work_dir}
echo save results to ${work_dir}

# training
python nmt.py \
    train \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --input-feed \
    --valid-niter 3000 \
    --batch-size 64 \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --label-smoothing 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --save-to ${work_dir}/model.bin \
    --lr-decay 0.5 

# decoding
python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
