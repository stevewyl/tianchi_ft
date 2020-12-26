TASKS=("ocemotion" "ocnli" "tnews")
for task_name in ${TASKS[@]}; do
    case $task_name in
    ocnli) bash run_classifier.sh ocnli 80 96 ;;
    ocemotion) bash run_classifier.sh ocemotion 192 32 ;;
    tnews) bash run_classifier.sh tnews 48 160 ;;
    esac
done