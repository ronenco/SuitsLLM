# SuitsLLM
A repository to generate a law answer ranker



# Generation of Law Advisor:

These are the steps taken when generating a new law advisor:

 ┌───────────────────────────┐
 │ 1. Load JSONL Dataset     │
 └──────────────┬────────────┘
                │  (LawExample objects)
                ▼
 ┌───────────────────────────┐
 │ 2. Split by Question      │
 │   - Train / Val / Test    │
 └──────────────┬────────────┘
                │  (3 example lists)
                ▼
 ┌───────────────────────────┐
 │ 3. Build HF DatasetDict   │
 │   - create "text" input   │
 │   - create float "label"  │
 └──────────────┬────────────┘
                │  (DatasetDict: train/val/test)
                ▼
 ┌───────────────────────────┐
 │ 4. Tokenization           │
 │   - pad/truncate          │
 │   - max_length            │
 └──────────────┬────────────┘
                │  (tokenized tensors)
                ▼
 ┌───────────────────────────┐
 │ 5. Load Base Model        │
 │   e.g. distilroberta-base │
 │   add regression head     │
 │   (optionally checkpoint) │
 └──────────────┬────────────┘
                │  (torch model)
                ▼
 ┌───────────────────────────┐
 │ 6. Trainer Setup          │
 │   - batch sizes           │
 │   - lr, epochs            │
 │   - grad accumulation     │
 │   - compute_metrics()     │
 └──────────────┬────────────┘
                │
                ▼
 ┌───────────────────────────┐
 │ 7. Training Loop          │
 │   - forward pass          │
 │   - loss (MSE)            │
 │   - backward → update     │
 │   - checkpointing enabled │
 └──────────────┬────────────┘
                │
                ▼
 ┌───────────────────────────┐
 │ 8. Evaluation (Val/Test)  │
 │   - MAE, RMSE, Pearson    │
 │   - Spearman              │
 └──────────────┬────────────┘
                │
                ▼
 ┌───────────────────────────┐
 │ 9. Save Final Model       │
 │   - tokenizer             │
 │   - model weights         │
 └───────────────────────────┘


 