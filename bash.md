# Bash Command Reference

Commands currently in use for the Subtask 3 pipeline. Each entry explains the purpose and the primary outputs or side effects.

## Docker Environment
- `cd subtask3`
  Go to folder.

- `docker compose --compatibility up -d`  
  Starts the CUDA-enabled container attached to the custom `dimabsa_net` bridge network. The workspace directory is mounted at `/workspace` in the container.

- `docker exec -it nlp_dimabsa_subtask3 /bin/bash`  
  Opens an interactive shell inside the running container so GPU jobs and scripts can be launched from `/workspace`.

- `nvidia-smi` *(run inside the container)*  
  Confirms that all four RTX 4090 GPUs are visible before training or inference.

## Model Training

### data laptop bert-base-multilingual-cased
- `cd starter_kit/task2task3 && CUDA_VISIBLE_DEVICES=1 python run_task2\&3_trainer_multilingual.py --task 3 --domain lap --language zho --train_data zho_laptop_train_alltasks.jsonl --infer_data zho_laptop_dev_task3.jsonl --bert_model_type bert-base-multilingual-cased --mode train --gpu True --epoch_num 1 --batch_size 1 --learning_rate 1e-3 --tuning_bert_rate 1e-5 --model_name zho_lap_subtask3`  
  Fine-tunes the Subtask 3 model on the laptop/ZH dataset using GPU 1. Produces:
  - `model/task3_lap_zho.pth` – latest checkpoint.  
  - `log/zho_lap_subtask3.log` – training losses and evaluation metrics.  
  - `tasks/subtask_3/pred_zho_laptop.jsonl` – dev predictions generated automatically at the end of training.

### data laptop bert-base-chinese
- `cd starter_kit/task2task3 && CUDA_VISIBLE_DEVICES=1 python run_task2\&3_trainer_multilingual.py --task 3 --domain lap --language zho --train_data zho_laptop_train_alltasks.jsonl --infer_data zho_laptop_dev_task3.jsonl --bert_model_type bert-base-chinese --mode train --gpu True --epoch_num 1 --batch_size 1 --learning_rate 1e-3 --tuning_bert_rate 1e-5 --model_name zho_lap_subtask3_bertzh`  
  Re-runs training with the Chinese-only BERT backbone. Outputs:
  - `model/task3_lap_zho.pth` (overwritten unless you change the model name).
  - `log/zho_lap_subtask3_bertzh.log` (if the model name differs).  
  - `tasks/subtask_3/pred_zho_laptop.jsonl` – refreshed predictions based on the new backbone.

### data restaurant bert-base-chinese
```bash
cd starter_kit/task2task3 && CUDA_VISIBLE_DEVICES=1 python run_task2\&3_trainer_multilingual.py \
  --task 3 \
  --domain res \
  --language zho \
  --train_data zho_restaurant_train_alltasks.jsonl \
  --infer_data zho_restaurant_dev_task3.jsonl \
  --bert_model_type bert-base-chinese \
  --mode train \
  --gpu True \
  --epoch_num 1 \
  --batch_size 1 \
  --learning_rate 1e-3 \
  --tuning_bert_rate 1e-5 \
  --model_name res_zho_subtask3_bertzh
```
- 這個腳本儲存 checkpoint 的方式是固定的：`model/task3_<domain>_<language>.pth`。你現在是 `--domain lap --language zho`，所以會產生 `model/task3_lap_zho.pth`；如果改成餐廳資料，只要把 `--domain` 換成 `res`，checkpoint 就會落在新的位置 `model/task3_res_zho.pth`，不會覆蓋掉原本的 laptop 模型。
- 對應地，輸出 JSON 也會用 `domain+language` 判斷檔名。餐廳/中文的預測會寫成 `tasks/subtask_3/pred_zho_restaurant.jsonl`，跟 `pred_zho_laptop.jsonl` 分開。
- Log 檔案則是用 `--model_name` 組出路徑 (`log/<model_name>.log`)；你只要給不同的名稱（例如 `res_zho_subtask3_bertzh`），就能保留兩份訓練記錄。

## Inference
- `CUDA_VISIBLE_DEVICES=1 python starter_kit/task2&3_trainer_multilingual.py --task 3 --domain lap --language zho --train_data zho_laptop_train_alltasks.jsonl --infer_data zho_laptop_dev_task3.jsonl --bert_model_type bert-base-multilingual-cased --mode inference --gpu True --inference_beta 0.9 --model_name zho_lap_subtask3 --reload True`  
  Reloads `model/task3_lap_zho.pth` and regenerates predictions for the dev file. Outputs overwrite:
  - `tasks/subtask_3/pred_zho_laptop.jsonl` (Quadruplet results).  

## Evaluation
- `python evaluation_script/metrics_subtask_1_2_3.py -t 3 -p tasks/subtask_3/pred_zho_laptop.jsonl -g evaluation_script/'sample data'/subtask_3/zho/gold_zho_laptop.jsonl`  
  Runs the official scorer on the sample ZH gold file (20 samples) to verify format and pipeline. Outputs precision/recall/F1 to stdout. Replace `-g` with the official dev gold (e.g., `data/zho_laptop_dev_task3_gold.jsonl`) for full evaluation when that file is available.
