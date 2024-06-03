#!/bin/bash

## kg-blip-dual-new-ao-all
# run.sh main_kg_blip_dual_new.py train kg-blip-dual-new-ao-all cuda_idx /path2kg /path2config not_loss_only seed n_epoch 20 sft
# run.sh main_kg_blip_dual_new.py valid kg-blip-dual-new-ao-all cuda_idx /path2kg /path2config loss_only seed epoch start end step
# run.sh main_kg_blip_dual_new.py test kg-blip-dual-new-ao-all cuda_idx /path2kg /path2config not_loss_only seed epoch archive_idx ...

# main_**.py
python_file=${1}

# train/test
running_type=${2}

# accumulate-base-kg/...
dir_name=${3}

# 1/2/3/4/...
device=${4}

# /path2EGMS/data/conceptnet/new/KG-10-AND-EXT-5-OR ...
kg_dir=${5}

# /path2EGMS/configs/accumulate-base-wo-kl.json ...
config_file=${6}

# loss_only/not_loss_only
loss_only=${7}

# 42
seed=${8}

# # when train: max epoch num (1/2/3)
# # when valid: epoch index (0/1/2)
# epoch=${8}

if [ $running_type == "train" ]
then
    epoch=${9}
    tend=${10}
    stage=${11}
    
    if [ $stage == "sft" ]
    then
        init_archive_file="/path2archive/saved_model/kg-blip-mm-ao-wo-kl/data20_epoch0.bin"
    elif [ $stage == "mm" ]
    then
        init_archive_file="/path2pretrained/pretrained-model/brio-cnndm-uncased/pytorch_model.bin"
    fi

    python $python_file \
    --init-embed \
    --archive-file $init_archive_file \
    --save-dir ./saved_model/$dir_name \
    --kg-dir $kg_dir \
    --data-set data1 \
    --config-file $config_file \
    --device cuda:$device \
    --modal-match $stage \
    --loss-only $loss_only \
    --cur-epoch 0 \
    --seed ${seed}
    # echo $stage
    
    for ((i=2; i<=$tend; i++))
    do
        python $python_file \
        --archive-file ./saved_model/$dir_name/data$(($i - 1))_epoch0.bin \
        --save-dir ./saved_model/$dir_name \
        --kg-dir $kg_dir \
        --data-set data$i \
        --config-file $config_file \
        --device cuda:$device \
        --modal-match $stage \
        --loss-only $loss_only \
        --cur-epoch 0 \
        --seed ${seed}
        # echo $stage $i
    done

    for ((i=2; i<=$epoch; i++))
    do
        python $python_file \
        --archive-file ./saved_model/$dir_name/data$(($tend))_epoch$(($i - 2)).bin \
        --save-dir ./saved_model/$dir_name \
        --kg-dir $kg_dir \
        --data-set data1 \
        --config-file $config_file \
        --device cuda:$device \
        --modal-match $stage \
        --loss-only $loss_only \
        --cur-epoch $(($i - 1)) \
        --seed ${seed}
        # echo $stage
        
        for ((j=2; j<=$tend; j++))
        do
            python $python_file \
            --archive-file ./saved_model/$dir_name/data$(($j - 1))_epoch$(($i - 1)).bin \
            --save-dir ./saved_model/$dir_name \
            --kg-dir $kg_dir \
            --data-set data$j \
            --config-file $config_file \
            --device cuda:$device \
            --modal-match $stage \
            --loss-only $loss_only \
            --cur-epoch $(($i - 1)) \
            --seed ${seed}
            # echo $stage $i
        done
    done
    
elif [ $running_type == "test" ]
then
    shift 8

    while [ $# -gt 0 ]
    do
        epoch=${1}
        data=${2}

        python $python_file \
        --test-only \
        --archive-file ./saved_model/$dir_name/data${data}_epoch${epoch}.bin \
        --save-dir ./saved_model/$dir_name-test \
        --kg-dir $kg_dir \
        --data-set test_data \
        --config-file $config_file \
        --device cuda:$device \
        --loss-only $loss_only \
        --seed ${seed}
        # echo $running_type $i

        shift 2
    done
    
elif [ $running_type == "valid" ]
then
    epoch=${9}
    start=${10}
    end=${11}
    step=${12}
    
    for ((i=$start; i<=$end; i=i+$step))
    do
        python $python_file \
        --valid-only \
        --archive-file ./saved_model/$dir_name/data$(($i))_epoch$(($epoch)).bin \
        --save-dir ./saved_model/$dir_name-valid \
        --kg-dir $kg_dir \
        --data-set valid_data \
        --config-file $config_file \
        --device cuda:$device \
        --loss-only $loss_only \
        --seed ${seed}
        # echo $running_type $i
    done

elif [ $running_type == "human_eval" ]
then
    shift 8

    while [ $# -gt 0 ]
    do
        epoch=${1}
        data=${2}

        python $python_file \
        --test-only \
        --archive-file ./saved_model/$dir_name/data${data}_epoch${epoch}.bin \
        --save-dir ./saved_model/$dir_name-human-eval \
        --kg-dir $kg_dir \
        --data-set human_eval_data \
        --config-file $config_file \
        --device cuda:$device \
        --loss-only $loss_only \
        --seed ${seed}
        # echo $running_type $i

        shift 2
    done

elif [ $running_type == "sample_eval" ]
then
    shift 8

    while [ $# -gt 0 ]
    do
        epoch=${1}
        data=${2}

        python $python_file \
        --test-only \
        --archive-file ./saved_model/$dir_name/data${data}_epoch${epoch}.bin \
        --save-dir ./saved_model/$dir_name-sample-eval \
        --kg-dir $kg_dir \
        --data-set sample_data \
        --config-file $config_file \
        --device cuda:$device \
        --loss-only $loss_only \
        --seed ${seed}
        # echo $running_type $i

        shift 2
    done
fi
