#!/bin/bash
SAVING_FOLDER="./checkpoint"       

pretrained=0
full_precision=0
recover=0
no_train=0
batch_size=512
learning_rate=0.001
num_test=0
device_id=-1
precisions=(3 4 5 6 7 8 9 10 11 12)

has_argument() {
    [[ ("$1" == *=* && -n ${1#*=}) || ( ! -z "$2" && "$2" != -*)  ]];
}

extract_argument() {
    echo "${2:-${1#*=}}"
}

# Function to handle options and arguments
handle_options() {
    while [ $# -gt 0 ]; do
        case $1 in
            -h | --help)
                return
                ;;
            -c | --config)
                if has_argument $@; then
                    config=$(extract_argument $@)
                    echo "Path to the config file: $config"
                    shift
                fi
                ;;
            --no_train)
                no_train=1
                ;;
            --pretrained)
                pretrained=1
                ;;
            --full_precision)
                full_precision=1
                ;;
            --recover)
                recover=1
                ;;
            --num_test)
                if has_argument $@; then
                    num_test=$(extract_argument $@)
                    echo "Number of test per model: $num_test"
                    shift
                fi
                ;;
            --bs | --batch_size)
                if has_argument $@; then
                    batch_size=$(extract_argument $@)
                    echo "batch size: $batch_size"
                    shift
                fi
                ;;
            --lr | --learning_rate)
                if has_argument $@; then
                    learning_rate=$(extract_argument $@)
                    echo "learning rate: $learning_rate"
                    shift
                fi
                ;;
            --device_id | --gpu)
                if has_argument $@; then
                    device_id=$(extract_argument $@)
                    echo "Device on which the train will be performed: $device_id"
                    shift
                fi
                ;; 
            *)
                echo "Invalid option: $1" >&2
                return 0
                ;;
        esac
        shift
    done
}

run_train() {
    saving_folder="$SAVING_FOLDER/bs$batch_size"_lr$learning_rate/
    mkdir log
    pids=()
    for i in $(eval echo "{1..$num_test}")
    do
        echo ""
        echo " BATCH SIZE $batch_size - LEARNING_RATE $learning_rate - PRECISION $precision - test $i "
        echo ""

        # construct the command
        cmd="python train.py \
            --saving_folder "$saving_folder" \
            --config $config \
            --batch_size $batch_size \
            --precision $precision \
            --lr $learning_rate \
            --experiment $i"

        if [ "$pretrained" -eq 1 ]; then
            echo "Loading the pretrained version"
            cmd="$cmd --pretrained"
        fi

        if [ "$no_train" -eq 1 ]; then
            echo "Run without training"
            cmd="$cmd --no_train"
        fi

        if [ "$recover" -eq 1 ]; then
            echo "Recover training from checkpoint"
            cmd="$cmd --recover"
        fi

        if [ "$device_id" -ne -1 ]; then
            echo "Device specified: $device_id"
            export CUDA_VISIBLE_DEVICES=$device_id
        fi

        $cmd > "./log/log_$precision"_"$i.txt" 2>&1 &

        pids+=($!)
        echo ""
        echo "-----------------------------------------------------------"
    done

    # wait for all background processes to finish
    for pid in "${pids[@]}"; do
        wait $pid
        current_date_time=$(date '+%Y-%m-%d %H:%M:%S')
        echo "$current_date_time: Process with PID $pid finished"
    done
}

# Main script execution
handle_options "$@"

#iterate over the precision
if [ "$full_precision" -eq 1 ]; then
    precision=32
    run_train
else
    for precision in ${precisions[*]}
    do
        run_train
    done
fi

return 0