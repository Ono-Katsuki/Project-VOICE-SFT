#!/usr/bin/env python3
import json, sys
from datetime import datetime
from macnotesapp import NotesApp

def iso(dt):
    if isinstance(dt, datetime):
        return dt.isoformat(timespec="seconds")
    return dt

def main():
    notes = NotesApp().notes()  # すべてのノート
    for n in notes:
        d = n.asdict()
        # 日付を ISO 文字列化
        d["creation_date"] = iso(d.get("creation_date"))
        d["modification_date"] = iso(d.get("modification_date"))
        # JSONL で 1 行ずつ出力（本文は HTML と plaintext の両方が入る）
        print(json.dumps(d, ensure_ascii=False))
if __name__ == "__main__":
    main()
