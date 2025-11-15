# DimABSAModel.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel

class DimABSA(nn.Module):
    def __init__(self, hidden_size, backbone_name, category_num, 
                 use_causal_lm=True, freeze_encoder=True):
        super().__init__()

        # 1) 載入 backbone：Qwen3-8B 用 CausalLM，BERT 可以走 AutoModel
        if use_causal_lm:
            self.encoder = AutoModelForCausalLM.from_pretrained(
                backbone_name,
                torch_dtype="auto",   # A100 80GB 可以讓它自動選 bfloat16 / fp16
            )
        else:
            self.encoder = AutoModel.from_pretrained(
                backbone_name,
                torch_dtype="auto",
            )

        # 2) 直接用模型 config 的 hidden_size
        self.hidden_size = self.encoder.config.hidden_size

        # 3) 判斷有沒有 token_type_ids（BERT 有，Qwen3 通常沒有）
        cfg_dict = self.encoder.config.to_dict()
        self.has_token_type_ids = cfg_dict.get("type_vocab_size", 0) > 0

        # 4) 是否凍結 encoder（先建議凍結，確認 pipeline 沒問題）
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # 5) 你的各種 head（下面名稱你可以換成你原本的）
        # span 任務：start / end 都是 [B, L, 2]
        self.a_start = nn.Linear(self.hidden_size, 2)
        self.a_end   = nn.Linear(self.hidden_size, 2)

        self.o_start = nn.Linear(self.hidden_size, 2)
        self.o_end   = nn.Linear(self.hidden_size, 2)

        self.ao_start = nn.Linear(self.hidden_size, 2)
        self.ao_end   = nn.Linear(self.hidden_size, 2)

        self.oa_start = nn.Linear(self.hidden_size, 2)
        self.oa_end   = nn.Linear(self.hidden_size, 2)

        # 類別分類（Task 3）
        self.fc_category = nn.Linear(self.hidden_size, category_num)

        # Valence / Arousal regression
        self.fc_valence  = nn.Linear(self.hidden_size, 1)
        self.fc_arousal  = nn.Linear(self.hidden_size, 1)

        # 如果真的要省記憶體，可以開：
        # self.encoder.gradient_checkpointing_enable()

    def encode(self, input_ids, attention_mask, token_type_ids):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }
        if self.has_token_type_ids and token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        if self.freeze_encoder:
            with torch.no_grad():
                outputs = self.encoder(**kwargs)
        else:
            outputs = self.encoder(**kwargs)

        # Qwen3 CausalLM 回傳的是 CausalLMOutputWithPast，
        # hidden_states[-1] 是最後一層 hidden: [B, L, H]
        sequence_output = outputs.hidden_states[-1]
        return sequence_output

    def forward(self, input_ids, attention_mask, token_type_ids, task_type):
        x = self.encode(input_ids, attention_mask, token_type_ids)

        if task_type == 'A':
            # aspect span
            start = self.a_start(x)  # [B, L, 2]
            end   = self.a_end(x)
            return start, end

        elif task_type == 'O':
            start = self.o_start(x)
            end   = self.o_end(x)
            return start, end

        elif task_type == 'AO':
            start = self.ao_start(x)
            end   = self.ao_end(x)
            return start, end

        elif task_type == 'OA':
            start = self.oa_start(x)
            end   = self.oa_end(x)
            return start, end

        elif task_type == 'C':
            # 類別分類：這裡用 CLS 或平均都可以，看你原本怎麼做
            pooled = x[:, 0, :]
            return self.fc_category(pooled)

        elif task_type == 'Valence':
            pooled = x[:, 0, :]
            return self.fc_valence(pooled).squeeze(-1)  # [B]

        elif task_type == 'Arousal':
            pooled = x[:, 0, :]
            return self.fc_arousal(pooled).squeeze(-1)  # [B]

        else:
            raise ValueError(f"Unknown task_type: {task_type}")
