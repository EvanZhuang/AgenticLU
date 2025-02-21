set -x

read -r -d '' training_commands <<EOF
openrlhf/cli/train_sft.py \
   --save_path ./checkpoint/llama3-8b-sft \
   --ckpt_path ./checkpoint/llama3-8b-sft \
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
   --dataset yzhuang/Agentic-Long-Context-Understanding-QA \
   --apply_chat_template \
   --input_key prompt \
   --output_key chosen \
   --ring_attn_size 8 \
   --ring_head_stride 2 \
   --packing_samples \
   --flash_attn \
   --overlap_comm \
   --gradient_checkpointing \
   --use_wandb True
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --ipo [for IPO]
    # --label_smoothing 0.1 [for cDPO]
    # --ref_offload
    # --packing_samples
    # --nll_loss_coef (Regularization with NLL loss)

if [[ ${1} != "slurm" ]]; then
    PYTHONPATH="../":"$PYTHONPATH" deepspeed --hostfile ./bash_scripts/hostfile [MAKE A HOST FILE YOURSELF] --master_addr [YOUR IP ADDRESS OF THE HEAD NODE] ${training_commands}
fi