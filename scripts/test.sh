#!/bin/bash
SAVING_FOLDER="./checkpoint"    

# Default variable values
batch_size=8
learning_rate=0.0015625
full_precision=0
pretrained=0
config=""
max_processes=1
num_models=-1
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
            --config | -c)
                if has_argument $@; then
                    config=$(extract_argument $@)
                    echo "Path to the config file: $config"
                    shift
                fi
                ;;
            --bs | --batch_size)
                if has_argument $@; then
                    batch_size=$(extract_argument $@)
                    echo "Batch size: $batch_size"
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
            --full_precision)
                full_precision=1
                echo "We will test the full precision model."
                ;;
            --device_id | --gpu)
                if has_argument $@; then
                    device_id=$(extract_argument $@)
                    echo "Device on which the train will be performed: $device_id"
                    shift
                fi
                ;; 
            --max_processes)
                if has_argument $@; then
                    max_processes=$(extract_argument $@)
                    echo "Number of processes in parallel: $max_processes"
                    shift
                fi
                ;;
            --num_models)
                if has_argument $@; then
                    num_models=$(extract_argument $@)
                    echo "Number of model to test: $num_models"
                    shift
                fi
                ;;
            *)
                echo "Invalid option: $1" >&2
                return
                ;;
        esac
        shift
    done
}


run_test() {
    if [ "$full_precision" -eq 1 ]; then
        precision=32
    fi
    saving_folder="$SAVING_FOLDER/bs$batch_size"_lr"$learning_rate/"
    mkdir log
    echo ""
    echo " BATCH SIZE $batch_size - LEARNING_RATE $learning_rate - PRECISION $precision - test $i "
    echo ""

    if [ "$device_id" -ne -1 ]; then
        echo "Device specified: $device_id"
        export CUDA_VISIBLE_DEVICES=$device_id
    fi

    cmd="python test.py \
        --saving_folder "$saving_folder" \
        --config $config \
        --batch_size $batch_size \
        --precision $precision \
        --lr $learning_rate \
        --num_models $num_models"

    $cmd > "./log/log_test_$precision.txt" 2>&1 &

    echo ""
    echo "-----------------------------------------------------------"

}


# Main script execution
handle_options "$@"
active_processes=0
# iterate over the precision
if [ "$full_precision" -eq 1 ]; then
    precision=32
    run_test
else
    for precision in ${precisions[*]}
    do
        run_test
        ((active_processes++))

        if [ "$active_processes" -ge "$max_processes" ]; then
            wait -n  # wait for at least one process to finish
            ((active_processes--))
        fi
    done
fi


