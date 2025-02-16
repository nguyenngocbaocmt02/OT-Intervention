TRAIN_DATASET="toxic"
SAVE="YOUR_SAVE_FOLDER"
##################################################
#################   Ours   #######################
##################################################  
# HF_NAMES = {llama_7B'
# 'alpaca_7B', 'vicuna_7B', 'llama2_chat_13B', 'llama3_8B' gpt2_large
#API='AIzaSyAsQRv2H_gvy9nCaDQWitQ_eRCyquIOoZA'
#API='AIzaSyBaxX9YnjZ3oW6zCZn5sXCF2BfrEClgczc'
API='AIzaSyC9PqIR1PvfGCjSjhPOnLpq_r-tUyh-AGU'
#API='AIzaSyCzJl-UkO-htWSeyDmE911JKk7jlneqXAk'
# TEST
for MODEL in gpt2_large; do
    for alpha in 40; do
        for bl in 2.5; do
            echo "model: $MODEL alpha: $alpha bl: $bl"
            CUDA_VISIBLE_DEVICES=3 python ot_edit_layer.py --api $API --clf_only 0 --exp test5 --model_name $MODEL --bl $bl --alpha $alpha --device 0 --train_dataset $TRAIN_DATASET --clf_folder $SAVE
            echo
            echo
        done
    done
done

##################################################
#################   ARGS    ######################
##################################################
# llama_7B --use_mode val --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass --judge_name --info_name --eval_dataset truthful_qa --train_dataset truthful_qa
