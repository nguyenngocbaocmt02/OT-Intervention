MODEL=llama_7B

<<<<<<< HEAD
# JUDGE=ft:davinci-002:ethicalytics::9t5UpZFx  ## NQOpen
JUDGE=ft:davinci-002:ethicalytics:truthful:A0WsrZ0l ## Truthful
INFO=ft:davinci-002:ethicalytics:informative:A0WuCDTp
LOG_FILE="sweep_output.log"
EVAL_DATASET="truthful_qa"
TRAIN_DATASET="truthful_qa"
SAVE="/home/users/nus/binhnt/scratch/baonn/clf"
CACHE="/home/users/nus/binhnt/scratch/.cache/huggingface/hub"
=======
JUDGE=
INFO=
LOG_FILE="sweep_output.log"
EVAL_DATASET="truthful_qa"
TRAIN_DATASET="truthful_qa"
SAVE=
>>>>>>> 726d78c57e286c0baeb03c57c1b5e4b332f140be
##################################################
#################   Ours   #######################
##################################################  
# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B'
# TEST
for MODEL in llama2_chat_13B_lofit_fold_0; do
    for alpha in 5.0; do
        for bl in 2.5; do
            echo "model: $MODEL alpha: $alpha bl: $bl"
            CUDA_VISIBLE_DEVICES=0 python ot_edit_layer.py --instruction_prompt default --exp_mode test --clf_only 0 --exp test4 --model_name $MODEL --bl $bl --alpha $alpha --device 0 --num_fold 2 --judge_name $JUDGE --info_name $INFO --eval_dataset $EVAL_DATASET --train_dataset $TRAIN_DATASET --clf_folder $SAVE --cache_dir $CACHE
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
# llama_7B --use_mode val --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name --info_name --eval_dataset truthful_qa --train_dataset truthful_qa
>>>>>>> 726d78c57e286c0baeb03c57c1b5e4b332f140be
