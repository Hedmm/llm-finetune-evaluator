r"""Build the 300-message Banking77 slice and, optionally, the replay predictions.

Banking77 (Casanueva et al., 2020) is by PolyAI and licensed CC BY 4.0. Its
test split is ``banking_data/test.csv`` in
https://github.com/PolyAI-LDN/task-specific-datasets (3,080 messages, 40 per
intent). The case-study repository ships the same file, byte for byte.

    python examples/banking77/make_slice.py --test-csv path/to/test.csv \
        --case-study path/to/ml-case-studies/lora-vs-qlora-vs-full-finetuning

Outputs, next to this script:

* ``data/banking77_slice.jsonl``: 300 messages drawn with a fixed seed, each
  tagged with a coarse intent ``group`` and a message ``length`` bucket.
* ``labels.txt``: the 77 intents, sorted as in the case study's prompt.
* ``predictions/<arm>.jsonl`` (with ``--case-study``): the recorded predictions
  of the case study's base, LoRA, QLoRA and full fine-tuning runs for the same
  300 messages, with the case study's own correctness flag for cross-checking.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
SAMPLE_SEED = 77
SAMPLE_SIZE = 300
ARMS = ("base", "lora", "qlora", "full")

GROUPS: dict[str, tuple[str, ...]] = {
    "cards": (
        "activate_my_card",
        "apple_pay_or_google_pay",
        "card_about_to_expire",
        "card_acceptance",
        "card_arrival",
        "card_delivery_estimate",
        "card_linking",
        "card_not_working",
        "card_swallowed",
        "change_pin",
        "compromised_card",
        "contactless_not_working",
        "disposable_card_limits",
        "get_disposable_virtual_card",
        "get_physical_card",
        "getting_spare_card",
        "getting_virtual_card",
        "lost_or_stolen_card",
        "order_physical_card",
        "pin_blocked",
        "supported_cards_and_currencies",
        "virtual_card_not_working",
        "visa_or_mastercard",
    ),
    "card_payments": (
        "Refund_not_showing_up",
        "card_payment_fee_charged",
        "card_payment_not_recognised",
        "card_payment_wrong_exchange_rate",
        "declined_card_payment",
        "direct_debit_payment_not_recognised",
        "extra_charge_on_statement",
        "pending_card_payment",
        "request_refund",
        "reverted_card_payment?",
        "transaction_charged_twice",
    ),
    "cash": (
        "atm_support",
        "cash_withdrawal_charge",
        "cash_withdrawal_not_recognised",
        "declined_cash_withdrawal",
        "pending_cash_withdrawal",
        "wrong_amount_of_cash_received",
        "wrong_exchange_rate_for_cash_withdrawal",
    ),
    "top_up": (
        "automatic_top_up",
        "balance_not_updated_after_cheque_or_cash_deposit",
        "pending_top_up",
        "top_up_by_bank_transfer_charge",
        "top_up_by_card_charge",
        "top_up_by_cash_or_cheque",
        "top_up_failed",
        "top_up_limits",
        "top_up_reverted",
        "topping_up_by_card",
        "verify_top_up",
    ),
    "transfers": (
        "balance_not_updated_after_bank_transfer",
        "beneficiary_not_allowed",
        "cancel_transfer",
        "declined_transfer",
        "failed_transfer",
        "pending_transfer",
        "receiving_money",
        "transfer_fee_charged",
        "transfer_into_account",
        "transfer_not_received_by_recipient",
        "transfer_timing",
    ),
    "exchange": ("exchange_charge", "exchange_rate", "exchange_via_app", "fiat_currency_support"),
    "account": (
        "age_limit",
        "country_support",
        "edit_personal_details",
        "lost_or_stolen_phone",
        "passcode_forgotten",
        "terminate_account",
        "unable_to_verify_identity",
        "verify_my_identity",
        "verify_source_of_funds",
        "why_verify_identity",
    ),
}
GROUP_OF = {intent: group for group, intents in GROUPS.items() for intent in intents}


def length_bucket(text: str) -> str:
    """Bucket a message by word count: short (up to 8), medium (9 to 16) or long."""
    words = len(text.split())
    if words <= 8:
        return "short"
    return "medium" if words <= 16 else "long"


def read_test_split(path: Path) -> list[tuple[str, str]]:
    """Read ``(text, intent)`` rows of the Banking77 test CSV."""
    with path.open(newline="", encoding="utf-8") as stream:
        return [(row["text"].strip(), row["category"].strip()) for row in csv.DictReader(stream)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSON lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def main() -> None:
    """Write the slice, the label list and (optionally) the replay predictions."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test-csv", type=Path, required=True, help="Banking77 test.csv")
    parser.add_argument("--case-study", type=Path, help="lora-vs-qlora-vs-full-finetuning folder")
    args = parser.parse_args()

    rows = read_test_split(args.test_csv)
    labels = sorted({intent for _, intent in rows})
    unmapped = sorted(set(labels) - set(GROUP_OF))
    if len(rows) != 3080 or len(labels) != 77 or unmapped:
        raise SystemExit(
            f"unexpected test split: {len(rows)} rows, {len(labels)} intents, unmapped {unmapped}"
        )
    chosen = sorted(random.Random(SAMPLE_SEED).sample(range(len(rows)), SAMPLE_SIZE))
    items = [
        {
            "id": f"b77-test-{index:04d}",
            "input": rows[index][0],
            "reference": rows[index][1],
            "task": "classification",
            "tags": {"group": GROUP_OF[rows[index][1]], "length": length_bucket(rows[index][0])},
            "metadata": {"split": "test", "row": index},
        }
        for index in chosen
    ]
    write_jsonl(HERE / "data" / "banking77_slice.jsonl", items)
    (HERE / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(items)} items covering {len({item['reference'] for item in items})} intents."
    )

    if args.case_study is None:
        return
    for arm in ARMS:
        lines = (args.case_study / "results" / arm / "preds.jsonl").read_text(encoding="utf-8")
        predictions = [json.loads(line) for line in lines.splitlines() if line.strip()]
        if len(predictions) != len(rows):
            raise SystemExit(f"{arm}: expected {len(rows)} predictions, found {len(predictions)}")
        exported = []
        for index in chosen:
            prediction = predictions[index]
            if prediction["text"] != rows[index][0] or prediction["gold"] != rows[index][1]:
                raise SystemExit(f"{arm}: prediction {index} does not match the test split")
            exported.append(
                {
                    "id": f"b77-test-{index:04d}",
                    "output": prediction["pred"],
                    "case_study_correct": prediction["correct"],
                }
            )
        write_jsonl(HERE / "predictions" / f"{arm}.jsonl", exported)
        print(f"Wrote predictions/{arm}.jsonl")


if __name__ == "__main__":
    main()
