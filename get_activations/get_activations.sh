for MODEL in qwen; do
    CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name $MODEL --dataset_name tqa_mc2 --train_dataset truthful_qa
#    CUDA_VISIBLE_DEVICES=0 python get_activations.py --model_name $MODEL --dataset_name tqa_gen_end_q --train_dataset truthful_qa
done
