# for task in "rag"; do
#     python eval.py --config configs/${task}.yaml --use_vllm
# done


# IDX=2
# for task in "icl"; do
#     CUDA_VISIBLE_DEVICES=$((2*IDX)),$((2*IDX+1)) python eval.py --config configs/${task}.yaml & 
#     IDX=$((IDX+1)) 
# done

# IDX=0
# for task in "rag" "summ" "cite" "longqa"; do
#     CUDA_VISIBLE_DEVICES=$((2*IDX)),$((2*IDX+1)) python eval.py --config configs/${task}.yaml & 
#     IDX=$((IDX+1)) 
# done


#this will run the 8k to 64k versions
# for task in "recall" "rag" "longqa" "summ" "icl" "rerank" "cite"; do
#     python eval.py --config configs/${task}_short.yaml
# done