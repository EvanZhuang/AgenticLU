set -x

read -r -d '' training_commands <<EOF
openrlhf/cli/train_dpo.py \
   --save_path ./checkpoint/llama3-8b-dpo \
   --ckpt_path ./checkpoint/llama3-8b-dpo \
   --max_ckpt_num 0 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Llama-3.1-8B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 131072 \
   --zero_stage 3 \
   --learning_rate 5e-7 \
   --beta 0.1 \
   --dataset yzhuang/Agentic-Long-Context-Understanding-QA \
   --apply_chat_template \
   --prompt_key prompt \
   --chosen_key chosen \
   --rejected_key rejected \
   --ring_attn_size 8 \
   --ring_head_stride 2 \
   --packing_samples \
   --flash_attn \
   --gradient_checkpointing \
   --adam_offload \
   --use_wandb True
EOF
    # --load_checkpoint \
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)
#CUDA_VISIBLE_DEVICES=0 

if [[ ${1} != "slurm" ]]; then
    PYTHONPATH="../":"$PYTHONPATH" deepspeed --hostfile ./bash_scripts/hostfile [MAKE A HOST FILE YOURSELF] --master_addr [YOUR IP ADDRESS OF THE HEAD NODE] ${training_commands}
fi