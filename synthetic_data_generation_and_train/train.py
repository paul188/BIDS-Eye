#!/usr/bin/env python3
"""
train.py — Fine-tune defog/sqlcoder-7b-2 on BIDS Text-to-SQL via QLoRA.

Model
─────
defog/sqlcoder-7b-2 is a Mistral-7B base fine-tuned by Defog specifically for
SQL generation.  It uses a structured prompt format with [SQL] / [/SQL] tags
rather than a chat template.  It does NOT respond in natural language — it
always outputs SQL.  This is intentional for a search-engine backend.

Prompt format (must match between training and inference)
─────────────────────────────────────────────────────────
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
...

### Database Schema
{SCHEMA_DDL}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
{sql}
[/SQL]

Loss function
─────────────
DataCollatorForCompletionOnlyLM with response_template="[SQL]\n".
Mistral's tokenizer encodes "[SQL]\n" as consistent tokens in-context, so
completion-only loss works correctly: only SQL tokens receive gradient signal.

Validation metrics
──────────────────
1. Syntax validity  (SV) — does the predicted SQL parse / execute without error?
2. Exact match      (EM) — normalised string equality (lowercase, collapsed whitespace)
3. Execution match  (EX) — result sets equal when run against the real DB (--db-url)

Usage
─────
    python train.py \\
        --data    ../training.jsonl \\
        --model   defog/sqlcoder-7b-2 \\
        --output  checkpoints/bids-sql-sqlcoder \\
        --db-url  "postgresql://user@localhost:5429/bids_sql" \\
        --epochs  3 \\
        --lora-r  16

Requirements
────────────
    pip install transformers peft trl datasets accelerate bitsandbytes sqlalchemy torch
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training_data_generation"))
from constants import SCHEMA_DDL  # noqa: E402


# ─── Prompt format ─────────────────────────────────────────────────────────────

RESPONSE_TEMPLATE = "[SQL]\n"
END_TOKEN         = "\n[/SQL]"

_INSTRUCTIONS = (
    "- Only use tables and columns present in the schema below.\n"
    "- Always SELECT: d.id, d.name, d.accession_id, d.bids_version, d.dataset_type, "
    "d.source_type, d.remote_url, d.validation_status, COUNT(DISTINCT o.subject) AS subject_count\n"
    "- Always: LEFT JOIN bids_objects o ON o.dataset_id = d.id AND o.subject IS NOT NULL\n"
    "- Always: GROUP BY d.id\n"
    "- Use EXISTS (...) subqueries to filter by file or participant properties\n"
    "- Use ILIKE '%%term%%' for case-insensitive text search\n"
    "- Default LIMIT 50 unless the question specifies otherwise\n"
    "- If the question cannot be answered from the schema, respond with:\n"
    "  SELECT 'I cannot determine the answer based on the given schema'"
)


def _prompt_prefix(question: str) -> str:
    """The fixed part before [SQL] — shared by training and inference."""
    return (
        f"### Task\n"
        f"Generate a SQL query to answer [QUESTION]{question}[/QUESTION]\n\n"
        f"### Instructions\n"
        f"{_INSTRUCTIONS}\n\n"
        f"### Database Schema\n"
        f"The query will run on a database with the following schema:\n"
        f"```sql\n{SCHEMA_DDL}\n```\n\n"
        f"### Answer\n"
        f"Given the database schema, here is the SQL query that "
        f"answers [QUESTION]{question}[/QUESTION]\n"
    )


def format_example(row: Dict) -> str:
    """Format one (question, sql) pair into a full training sequence."""
    return _prompt_prefix(row["input"].strip()) + RESPONSE_TEMPLATE + row["output"].strip() + END_TOKEN


def format_inference(question: str) -> str:
    """Format a question for inference — stops before [SQL] so model continues."""
    return _prompt_prefix(question.strip()) + RESPONSE_TEMPLATE


# ─── Data loading ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def make_dataset(rows: List[Dict], tokenizer) -> Dataset:
    texts = [format_example(r) for r in rows]
    return Dataset.from_dict({"text": texts})


# ─── Evaluation helpers ─────────────────────────────────────────────────────────

def normalise_sql(sql: str) -> str:
    sql = sql.lower().strip().rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def check_syntax(sql: str, db_url: Optional[str]) -> Tuple[bool, Optional[str]]:
    if db_url:
        from sqlalchemy import create_engine, text
        try:
            engine = create_engine(db_url)
            with engine.connect() as conn:
                conn.execute(text(f"EXPLAIN {sql}"))
            return True, None
        except Exception as e:
            return False, str(e)
    else:
        try:
            import sqlparse
            parsed = sqlparse.parse(sql)
            return bool(parsed and parsed[0].tokens), None
        except ImportError:
            return True, None


def execution_match(ref_sql: str, pred_sql: str, db_url: str) -> bool:
    from sqlalchemy import create_engine, text
    engine = create_engine(db_url)
    try:
        with engine.connect() as conn:
            ref_rows  = set(tuple(r) for r in conn.execute(text(ref_sql)).fetchall())
            pred_rows = set(tuple(r) for r in conn.execute(text(pred_sql)).fetchall())
        return ref_rows == pred_rows
    except Exception:
        return False


def _extract_sql_from_output(raw: str) -> str:
    """
    Extract SQL from the model's raw output.
    SQLCoder outputs SQL between [SQL] and [/SQL] tags (both may be absent if
    generation was cut short).  We also handle accidental markdown fences.
    """
    # Between [SQL] ... [/SQL]
    m = re.search(r"\[SQL\]\s*(.*?)(?:\[/SQL\]|$)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if re.match(r"(?i)select\b", candidate):
            return candidate

    # Markdown fence
    m = re.search(r"```(?:sql)?\s*(.*?)(?:```|$)", raw, re.DOTALL | re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if re.match(r"(?i)select\b", candidate):
            return candidate

    # Raw SQL at start
    stripped = raw.strip()
    if re.match(r"(?i)select\b", stripped):
        return stripped.split("\n\n")[0].strip()

    return ""


def evaluate(
    model,
    tokenizer,
    val_rows: List[Dict],
    db_url: Optional[str],
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> Dict[str, float]:
    model.eval()
    sv_ok = em_ok = ex_ok = 0
    sv_total = em_total = ex_total = 0

    for row in val_rows:
        prompt = format_inference(row["input"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        pred_sql = _extract_sql_from_output(generated)

        sv_total += 1
        ok, _ = check_syntax(pred_sql, db_url)
        if ok:
            sv_ok += 1

        em_total += 1
        if normalise_sql(pred_sql) == normalise_sql(row["output"]):
            em_ok += 1

        if db_url and ok:
            ex_total += 1
            if execution_match(row["output"], pred_sql, db_url):
                ex_ok += 1

    metrics = {
        "syntax_validity": sv_ok / sv_total if sv_total else 0.0,
        "exact_match":     em_ok / em_total if em_total else 0.0,
    }
    if ex_total:
        metrics["execution_match"] = ex_ok / ex_total
    return metrics


# ─── Training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",    type=Path, required=True)
    parser.add_argument("--model",   default="defog/sqlcoder-7b-2",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--output",  type=Path, default=Path("checkpoints/bids-sql-sqlcoder"))
    parser.add_argument("--db-url",  default=None)
    parser.add_argument("--epochs",  type=int, default=3)
    parser.add_argument("--batch",   type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr",      type=float, default=2e-4)
    parser.add_argument("--lora-r",  type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=1024,
                        help="Max token length — SQLCoder prompts are longer than Phi-3 due to DDL")
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model: {args.model}")

    # ── Load data ───────────────────────────────────────────────────────────────
    rows = load_jsonl(args.data)
    print(f"Loaded {len(rows)} examples from {args.data}")

    import random
    random.seed(args.seed)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_split))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    # ── Tokeniser ───────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model + QLoRA ───────────────────────────────────────────────────────────
    bnb_cfg = None
    if args.load_in_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.load_in_4bit else None,
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # SQLCoder is Mistral-based — standard attention + MLP projection names
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Completion-only loss ────────────────────────────────────────────────────
    # Mistral's tokenizer encodes "[SQL]\n" consistently in-context, so
    # DataCollatorForCompletionOnlyLM works correctly here (unlike Phi-3).
    # Only the SQL tokens (after [SQL]\n) receive gradient signal.
    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    # ── Training arguments ──────────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        logging_first_step=True,
        disable_tqdm=True,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )

    train_dataset = make_dataset(train_rows, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=args.max_len,
        packing=False,
    )

    print("\nStarting training...")
    trainer.train()
    trainer.save_model(str(args.output / "final"))
    print(f"Model saved to {args.output / 'final'}")

    print("\nRunning validation...")
    metrics = evaluate(
        model=model,
        tokenizer=tokenizer,
        val_rows=val_rows,
        db_url=args.db_url,
        device=device,
    )

    print("\n── Validation Results ──────────────────────────────────────────")
    print(f"  Syntax validity  (SV): {metrics['syntax_validity']:.1%}  "
          f"({int(metrics['syntax_validity']*len(val_rows))}/{len(val_rows)})")
    print(f"  Exact match      (EM): {metrics['exact_match']:.1%}")
    if "execution_match" in metrics:
        print(f"  Execution match  (EX): {metrics['execution_match']:.1%}  (gold standard)")
    print("────────────────────────────────────────────────────────────────\n")

    metrics_path = args.output / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
