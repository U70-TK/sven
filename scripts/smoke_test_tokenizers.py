"""Smoke test: load each prefix model end-to-end via the patched code path,
generate a few tokens, and verify decode quality. Run on GPU.

Reports per model: tokenizer class, first prompt token ids, decoded generation,
presence of corruption markers (literal Ġ / Ċ leaking into decoded text),
and whether spaces survived.
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sven.model import load_model

PROMPT = (
    "from flask import Flask, request, send_file, safe_join\n"
    "import os\n\n"
    "app = Flask(__name__)\n\n"
    '@app.route("/read")\n'
    "def read():\n"
    "    requested_file_name = request.args.get('filename')\n"
    "    safe_dir = '/safe/'\n\n"
    "    # read the requested file from the safe directory\n"
)


def make_args(device):
    a = argparse.Namespace()
    a.device = device
    a.n_gpu = 1
    return a


def test_model(name, ckpt_dir, device):
    print(f"\n=== {name} ===")
    args = make_args(device)
    try:
        tokenizer, model, input_device = load_model("prefix", ckpt_dir, False, args)
    except Exception as e:
        print(f"  LOAD ERROR: {type(e).__name__}: {e}")
        return

    print(f"  tokenizer class: {type(tokenizer).__name__}  is_fast={getattr(tokenizer, 'is_fast', None)}")
    enc = tokenizer(PROMPT, return_tensors="pt").to(input_device)
    ids = enc["input_ids"][0].tolist()
    print(f"  prompt first 10 tok ids: {ids[:10]}")
    print(f"  prompt first 10 tok strs: {tokenizer.convert_ids_to_tokens(ids[:10])}")

    # decode-only round-trip on the prompt
    dec_prompt = tokenizer.decode(ids, skip_special_tokens=True)
    rt = dec_prompt == PROMPT
    print(f"  prompt roundtrip ok: {rt}")
    if not rt:
        # show where it diverges
        for i, (a_, b_) in enumerate(zip(PROMPT, dec_prompt)):
            if a_ != b_:
                print(f"  first diff at char {i}: expected {a_!r}, got {b_!r}")
                break

    # short generation through the actual model + prefix
    model.eval()
    with torch.no_grad():
        # control_id=0 corresponds to "sec" prefix
        control_id = torch.tensor([0], device=input_device)
        try:
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                control_id=control_id,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        except TypeError:
            # some prefix models don't take control_id kwarg in generate
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

    new_ids = out[0, enc["input_ids"].shape[1]:].tolist()
    gen_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"  generated ids[:10]: {new_ids[:10]}")
    print(f"  generated text repr: {gen_text!r}")
    has_corruption = ("Ġ" in gen_text) or ("Ċ" in gen_text)
    has_space = " " in gen_text or "\n" in gen_text
    verdict = "CORRUPT" if has_corruption else ("CLEAN" if has_space else "SUSPICIOUS-NO-WS")
    print(f"  VERDICT: {verdict}")

    # free GPU memory before next model
    del model, tokenizer, out, enc
    torch.cuda.empty_cache()


def main():
    device = torch.device("cuda")
    print("device:", device, "name:", torch.cuda.get_device_name(0))

    trained_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained")
    models = sorted(
        d for d in os.listdir(trained_root)
        if d.endswith("-prefix") and os.path.isdir(os.path.join(trained_root, d, "checkpoint-last"))
    )
    for m in models:
        test_model(m, os.path.join(trained_root, m, "checkpoint-last"), device)


if __name__ == "__main__":
    main()
