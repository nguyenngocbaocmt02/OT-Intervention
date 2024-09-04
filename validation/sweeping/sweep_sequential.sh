MODEL=llama_7B
# INFO=ft:davinci-002:ethicalytics::9Rh9PRe7
# JUDGE=ft:davinci-002:ethicalytics::9RgySkJq
JUDGE=ft:davinci-002:ethicalytics:truthful:A0WsrZ0l
INFO=ft:davinci-002:ethicalytics:informative:A0WuCDTp
LOG_FILE="sweep_output.log"
EVAL_DATASET="truthful_qa"
TRAIN_DATASET="truthful_qa"
# CUDA_VISIBLE_DEVICES=2,3 python choose_edited_layer.py $MODEL --bl 1.0 --loss_type fpr_fnr
##################################################
#################   ITI    #######################
##################################################
# TEST
for MODEL in llama_7B alpaca_7B llama2_chat_7B vicuna_7B llama2_chat_13B llama3_8B; do
    for alpha in 15; do
        for K in 48; do
            echo "model: $MODEL alpha: $alpha K: $K"
            CUDA_VISIBLE_DEVICES=2,3 python validate_2fold.py --model_name $MODEL --use_mode test --num_heads $K --alpha $alpha --device 0 --num_fold 2 --use_center_of_mass --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET
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

##################################################
#################   ARGS    ######################
##################################################
# llama_7B --use_mode val --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name ft:davinci-002:ethicalytics::9Rh9PRe7 --info_name ft:davinci-002:ethicalytics::9RgySkJq --eval_dataset truthful_qa --train_dataset truthful_qa
