MODEL=llama_7B
<<<<<<< HEAD
JUDGE=ft:davinci-002:ethicalytics:truthful:A0WsrZ0l # TruthfulQA
#JUDGE=ft:davinci-002:ethicalytics::9t5UpZFx  ## NQOpen
#JUDGE=ft:davinci-002:ethicalytics::9t6np2Pi  ## TRivia
INFO=ft:davinci-002:ethicalytics:informative:A0WuCDTp
=======

JUDGE=
INFO=
>>>>>>> 726d78c57e286c0baeb03c57c1b5e4b332f140be
LOG_FILE="sweep_output.log"
EVAL_DATASET="truthful_qa"
TRAIN_DATASET="truthful_qa"
CACHE="/home/users/nus/binhnt/scratch/.cache/huggingface/hub"
# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B'
# TEST
##################################################
#################   ITI    #######################
##################################################
#TEST
for MODEL in llama3_8B_lofit_fold_0; do
    for alpha in 5 10; do
        for K in 48; do
            echo "model: $MODEL alpha: $alpha K: $K"
            CUDA_VISIBLE_DEVICES=0 python validate_2fold.py --instruction_prompt default --model_name $MODEL --use_mode test --num_heads $K --alpha $alpha --device 0 --num_fold 2 --use_center_of_mass --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET --cache_dir $CACHE
            echo
            echo
        done
    done
done

<<<<<<< HEAD

=======
##################################################
#################   ARGS    ######################
##################################################
# llama_7B --use_mode val --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name  --info_name --eval_dataset truthful_qa --train_dataset truthful_qa
>>>>>>> 726d78c57e286c0baeb03c57c1b5e4b332f140be
