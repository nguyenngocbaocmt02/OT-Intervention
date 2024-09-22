MODEL=llama_7B
TRAIN_DATASET="truthful_qa"
BL=1.0
LOSS_TYPE=cross_entropy

for MODEL in alpaca_7B; do
    echo "model: $MODEL"
    CUDA_VISIBLE_DEVICES=0 python choose_edited_layer.py --model_name $MODEL --bl $BL --loss_type $LOSS_TYPE --device 0 --num_fold 2 --train_dataset $TRAIN_DATASET
    echo
    echo
done