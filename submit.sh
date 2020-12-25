TIMESTAMP=$(date +%m%d)
RESULT_DIR=result/$TIMESTAMP
if [ ! -f $RESULT_DIR ]; then
    mkdir $RESULT_DIR
fi
cp data/ocemotion/ocemotion_predict.json .
cp data/ocnli/ocnli_predict.json .
cp data/tnews/tnews_predict.json .
zip $RESULT_DIR/submit.zip ocemotion_predict.json ocnli_predict.json tnews_predict.json
rm ./*.json