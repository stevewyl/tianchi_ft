TASK_NAME=$1
MODEL_NAME=${2:-"roberta"}
MAX_SEQ_LEN=${3:-128}
NUM_EPOCHS=${4:-5}
MODEL_NAME_FULL="hfl/chinese-"$MODEL_NAME"-wwm-ext"
# TIMESTAMP=$(date %m%d%H%M)
OUTPUT_DIR="model_result/"$TASK_NAME"_"$MODEL_NAME"_epoch"$NUM_EPOCHS

if [ -f $OUTPUT_DIR"/pytorch_model.bin" ]; then
    MODEL_NAME_FULL=$OUTPUT_DIR
fi

python run_classifier.py \
    --model_name $MODEL_NAME_FULL \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_predict \
    --data_dir data/$TASK_NAME \
    --max_seq_length $MAX_SEQ_LEN \
    --per_device_train_batch_size 48 \
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
    --metric_for_best_model "eval_f1"

rm -rf $OUTPUT_DIR/checkpoint-*
