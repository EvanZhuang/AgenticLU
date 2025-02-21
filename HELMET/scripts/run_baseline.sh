model="meta-llama/Llama-3.1-8B-Instruct"
base_model=$(echo $model | cut -d'/' -f 2)
task="rag"

PYTHONPATH="./":"$PYTHONPATH" python eval.py --config configs/${task}.yaml --use_vllm --model_name_or_path $model --output_dir output/${base_model} --shots 0 --generation_max_length 1024 --do_sample True --temperature 0.6