MODEL=llama_7B
TRAIN_DATASET="toxic"
# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B'
# TEST
API='AIzaSyAsQRv2H_gvy9nCaDQWitQ_eRCyquIOoZA'
# API='AIzaSyBaxX9YnjZ3oW6zCZn5sXCF2BfrEClgczc'
# API='AIzaSyC9PqIR1PvfGCjSjhPOnLpq_r-tUyh-AGU'
# API='AIzaSyCzJl-UkO-htWSeyDmE911JKk7jlneqXAk'
##################################################
#################   ITI    #######################
##################################################
#TEST
for MODEL in gpt2_large; do
    for alpha in 15; do
        for K in 15; do
            echo "model: $MODEL alpha: $alpha K: $K"
            CUDA_VISIBLE_DEVICES=3 python validate_2fold.py --api $API --model_name $MODEL --num_heads $K --alpha $alpha --device 0 --use_center_of_mass --train_dataset $TRAIN_DATASET
            echo
            echo
        done
    done
done

##################################################
#################   UNINTERVENED    ##############
##################################################
# for MODEL in llama_7B alpaca_7B llama2_chat_7B vicuna_7B llama2_chat_13B llama3_8B; do
#     echo "model : $MODEL"
#     CUDA_VISIBLE_DEVICES=0,1 python validate_2fold.py --model_name $MODEL --use_mode test --num_heads 0 --alpha 0 --device 0 --num_fold 2 --use_center_of_mass --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET
# done

#--model_name gpt2_large --use_mode test --num_heads 48 --alpha_ot 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name d --info_name d --eval_dataset truthful_qa --train_dataset truthful_qa
