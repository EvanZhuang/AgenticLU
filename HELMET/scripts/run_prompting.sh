model="yzhuang/Llama-3.1-8B-Instruct-AgenticLU"
base_model=$(echo $model | cut -d'/' -f 2)
task="rag"

for prompting_method in "stepbystep"  "plan&solve" "fact&reflect"; do 
    PYTHONPATH="./":"$PYTHONPATH" python eval_prompting.py --config configs/${task}.yaml --use_vllm --prompting_method $prompting_method --output_dir output/Llama-3.1-8B-Instruct-${prompting_method} --shots 0 --generation_max_length 1024
done