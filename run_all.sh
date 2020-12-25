TIMESTAMP=$(date +%m%d%H%M)

TASKS=("ocemotion" "ocnli" "tnews")
for task_name in ${TASKS[@]}; do
    LOG_FILE=log/$task_name"_"$TIMESTAMP".log"
    echo "[TASK] "$task_name" [LOG] "$LOG_FILE
    case $task_name in
    ocnli) bash run_classifier.sh ocnli 80 96 > $LOG_FILE
    ocemotion) bash run_classifier.sh ocemotion 192 32 > $LOG_FILE
    tnews) bash run_classifier.sh tnews 48 160 > $LOG_FILE
    esac