PYTHONPATH="./":"$PYTHONPATH" python ./long_context_llm/qa_tree_datagen.py \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --max_sample_size 8 \
    --max_tree_depth 2 \
    --dataset_name yzhuang/narrative_qa