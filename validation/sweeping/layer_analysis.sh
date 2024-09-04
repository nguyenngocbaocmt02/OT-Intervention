MODEL=llama_7B
TRAIN_DATASET="truthful_qa"
BL=1.0
LOSS_TYPE=fpr_fnr

for MODEL in llama_7B alpaca_7B llama2_chat_7B vicuna_7B llama2_chat_13B llama3_8B; do
    echo "model: $MODEL"
    CUDA_VISIBLE_DEVICES=2,3 python choose_edited_layer.py --model_name $MODEL --bl $BL --loss_type $LOSS_TYPE --device 0 --num_fold 2 --train_dataset $TRAIN_DATASET
    echo
    echo
done