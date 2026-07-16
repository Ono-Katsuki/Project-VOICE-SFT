#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
複数のJSONL(メール/メッセージ/ノート/スレッド)を1本にまとめて、
  {
    "id": 1,
    "text": "今日は〜",
    "source_file": "wheelhirai-threads_full.jsonl",
    "source_kind": "threads_full",
    "source_id": "msg-1234"   # あれば
  }
みたいな形にするやつ。

使い方:
  python dataset/merge_sources_to_base.py dataset/*.jsonl > dataset/base_merged.jsonl
"""
import sys, json, os
from pathlib import Path

def pick_text_from_obj(obj):
    # iMessage
    if "text" in obj and isinstance(obj.get("text"), str):
        return obj["text"].strip()
    # Gmail 1通ずつ
    for k in ("body_text", "body_html", "plaintext", "body"):
        if k in obj and isinstance(obj[k], str) and obj[k].strip():
            return obj[k].strip()
    return None

def iter_from_threads_full(obj):
    # {"messages":[...]} なやつ（GmailスレッドやiMessage風）をバラす
    messages = obj.get("messages") or []
    for m in messages:
        yield m

def main():
    files = sys.argv[1:]
    if not files:
        print("usage: merge_sources_to_base.py <jsonl...>", file=sys.stderr)
        sys.exit(1)

    out_id = 1
    for f in files:
        src_name = Path(f).name
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # 1) スレッド型なら中のメッセージを展開
                if "messages" in obj and isinstance(obj["messages"], list):
                    for m in iter_from_threads_full(obj):
                        # メール/メッセージはあなたが送信したものだけ
                        if m.get("direction") and m["direction"] != "outbound":
                            continue
                        txt = pick_text_from_obj(m)
                        if not txt:
                            continue
                        rec = {
                            "id": out_id,
                            "text": txt,
                            "source_file": src_name,
                            "source_kind": "threads_full",
                        }
                        # 元メッセージのIDがあれば付ける
                        for cid in ("id","message_id","message_guid","thread_id"):
                            if cid in m:
                                rec["source_id"] = str(m[cid])
                                break
                        print(json.dumps(rec, ensure_ascii=False))
                        out_id += 1
                    continue

                # 2) フラット1行=1メッセージ型（Gmail/Notes/iMessage）
                # メール・メッセージは送信だけ
                if obj.get("direction") and obj["direction"] != "outbound":
                    continue

                txt = pick_text_from_obj(obj)
                if not txt:
                    continue

                # ノートは基本自分が書いたものなので direction 無しでも通す
                source_kind = "generic"
                if "imessage" in src_name:
                    source_kind = "imessage"
                elif "notes" in src_name:
                    source_kind = "apple-notes"
                elif "gmail" in src_name or "messages" in src_name:
                    source_kind = "gmail"
                elif "threads_full" in src_name:
                    source_kind = "threads_full"

                rec = {
                    "id": out_id,
                    "text": txt,
                    "source_file": src_name,
                    "source_kind": source_kind,
                }
                for cid in ("id","message_id","message_guid","thread_id"):
                    if cid in obj:
                        rec["source_id"] = str(obj[cid])
                        break
                print(json.dumps(rec, ensure_ascii=False))
                out_id += 1

if __name__ == "__main__":
    main()
