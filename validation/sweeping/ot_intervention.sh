JUDGE=gpt-4
INFO=gpt-4
LOG_FILE="sweep_output.log"
EVAL_DATASET="truthful_qa"
TRAIN_DATASET="truthful_qa"
SAVE=save_info
##################################################
#################   Ours   #######################
##################################################  
# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B' gpt2_large
# TEST
for MODEL in mistral; do
    for alpha in 10.0; do
        for bl in 2.0; do
            echo "model: $MODEL alpha: $alpha bl: $bl"
            CUDA_VISIBLE_DEVICES=3 python ot_edit_layer.py --instruction_prompt default --exp_mode test --clf_only 0 --exp test4 --model_name $MODEL --bl $bl --alpha $alpha --device 0 --num_fold 2 --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET --clf_folder $SAVE
            echo
            echo
        done
    done
done

##################################################
#################   ARGS    ######################
##################################################
# llama_7B --use_mode val --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name --info_name --eval_dataset truthful_qa --train_dataset truthful_qa
