
for i in {0..5}; do
    python scripts/process_chatbot_arena.py ../lmsys_chat_1m_$i.parquet ../out_$i.parquet \
        --generation_tokens 192 --main_model meta-llama/Llama-2-13b-chat-hf --batch_size 64 \
        --target_model llama-2-13b-chat
done