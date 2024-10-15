
CACHE=""
for MODEL in llama2_chat_13B_lofit_fold_0 llama2_chat_13B_lofit_fold_1; do
    CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name $MODEL --cache_dir $CACHE --dataset_name tqa_mc2
    CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name $MODEL --cache_dir $CACHE --dataset_name tqa_gen_end_q
done
