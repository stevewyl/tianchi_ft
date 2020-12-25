TASK_NAME=$1
MAX_SEQ_LEN=${2:-128}
BATCH_SIZE=${3:-48}
MODEL_NAME=${4:-"roberta"}
NUM_EPOCHS=${5:-5}
MODEL_NAME_FULL="hfl/chinese-"$MODEL_NAME"-wwm-ext"
TIMESTAMP=$(date +%m%d%H%M)
OUTPUT_DIR="model_result/"$TASK_NAME"_"$MODEL_NAME"_epoch"$NUM_EPOCHS
LOG_FILE=log/$TASK_NAME"_"$TIMESTAMP".log"

if [ -f $OUTPUT_DIR"/pytorch_model.bin" ]; then
    MODEL_NAME_FULL=$OUTPUT_DIR
fi

nohup python run_classifier.py \
    --model_name $MODEL_NAME_FULL \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir data/$TASK_NAME \
    --max_seq_length $MAX_SEQ_LEN \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --warmup_steps 100 \
    --save_steps 300 \
    --eval_steps 100 \
    --save_total_limit 10 \
    --evaluation_strategy steps \
    --num_train_epochs $NUM_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --load_best_model_at_end \
    --metric_for_best_model "eval_f1" > $LOG_FILE 2>&1 &

rm -rf $OUTPUT_DIR/checkpoint-*
