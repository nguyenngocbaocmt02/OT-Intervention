for MODEL in gpt2_large; do
    CUDA_VISIBLE_DEVICES=3 python get_activations.py --model_name $MODEL --dataset_name tqa_mc2 --train_dataset toxic
    #CUDA_VISIBLE_DEVICES=1 python get_activations.py --model_name $MODEL --dataset_name tqa_gen_end_q --train_dataset toxic
done