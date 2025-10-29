#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macOS Messages (iMessage/SMS) -> JSONL exporter
- 標準ライブラリのみ（sqlite3, json など）
- 元DB (~/Library/Messages/chat.db) を安全にバックアップしてから参照
- 1) messages_flat.jsonl（1行=1メッセージ）
- 2) threads_full.jsonl（1行=1チャット、messages配列つき）
- 期間フィルタ: --since 'YYYY-MM-DD' / --since-days 180 / 全期間（指定なし）
- 方向: direction = outbound / inbound（is_from_me）
- 付属情報: chat_guid, participants, service (iMessage/SMS), attachments（パス・MIME など）

注意:
- DBスキーマはmacOSで多少差異あり（dateがns/秒）。normalize_date()で補正。
- 添付パスは ~/Library/Messages/Attachments/... を指す（実体はコピーしない）。
- 読み取りにはフルディスクアクセスが必要。
"""
import argparse, json, os, sqlite3, sys, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

APPLE_EPOCH_OFFSET = 978307200  # seconds from 1970-01-01 to 2001-01-01

def normalize_date(raw: Optional[int]) -> Optional[str]:
    """
    AppleのMessagesのdate系は環境により:
      - 2001-01-01基準の「秒」
      - 2001-01-01基準の「ナノ秒」
    が混在。ラフに補正してISO文字列を返す。
    """
    if raw is None:
        return None
    try:
        v = int(raw)
    except Exception:
        return None
    # ナノ秒っぽい大きさなら秒に直す
    if abs(v) > 3_000_000_000:  # ~95年相当を閾値に
        sec = v / 1_000_000_000
    else:
        sec = v
    ts = sec + APPLE_EPOCH_OFFSET
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.isoformat(timespec="seconds")
    except Exception:
        return None

def backup_sqlite(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # sqlite3 のバックアップAPIで安全コピー
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as src_conn:
        with sqlite3.connect(dst) as dst_conn:
            src_conn.backup(dst_conn)

def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return cols

def build_attachment_map(conn: sqlite3.Connection) -> Dict[int, List[Dict[str,Any]]]:
    cols = set(get_columns(conn, "attachment"))
    # filename と filepath のどちらがあるか
    prefer_path_col = "filename" if "filename" in cols else ("filepath" if "filepath" in cols else None)
    mime_col = "mime_type" if "mime_type" in cols else ("uti" if "uti" in cols else None)
    size_col = "total_bytes" if "total_bytes" in cols else None

    q = f"""
        SELECT maj.message_id,
               a.ROWID as attachment_id,
               {prefer_path_col if prefer_path_col else "NULL"} as path_like,
               {mime_col if mime_col else "NULL"} as mime_like,
               {size_col if size_col else "NULL"} as size_like,
               a.transfer_name
        FROM attachment a
        JOIN message_attachment_join maj ON a.ROWID = maj.attachment_id
    """
    amap: Dict[int, List[Dict[str,Any]]] = {}
    for row in conn.execute(q):
        mid = row[0]
        att = {
            "attachment_id": row[1],
            "path": row[2],
            "mime": row[3],
            "bytes": row[4],
            "name": row[5],
        }
        amap.setdefault(mid, []).append(att)
    return amap

def load_participants(conn: sqlite3.Connection) -> Dict[int, List[str]]:
    """
    chat_id -> [handle_id(str), ...]
    """
    q = """
        SELECT c.ROWID as chat_id, h.id as handle
        FROM chat c
        JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
        JOIN handle h ON h.ROWID = chj.handle_id
    """
    mapping: Dict[int, List[str]] = {}
    for chat_id, handle in conn.execute(q):
        mapping.setdefault(chat_id, []).append(handle)
    return mapping

def main():
    p = argparse.ArgumentParser(description="Export Messages.app (iMessage/SMS) to JSONL")
    home = Path.home()
    default_db = home / "Library/Messages/chat.db"
    here = Path(__file__).resolve().parent
    p.add_argument("--db", default=str(default_db), help="Source chat.db path")
    p.add_argument("--backup", default=str(here / "cache/chat.copy.sqlite"), help="Backup sqlite path")
    p.add_argument("--out", default=str(here / "imessage_messages.jsonl"), help="Flat messages JSONL")
    p.add_argument("--threads-out", default=str(here / "imessage_threads.jsonl"),
                   help="Threads JSONL (1行=1チャット＋messages配列)")
    # 期間フィルタ
    p.add_argument("--since", default=None, help="YYYY-MM-DD 以降のメッセージに限定")
    p.add_argument("--since-days", type=int, default=None, help="直近N日だけ")
    p.add_argument("--timezone", default="UTC", help="表記上の注意: 出力はUTC ISO。必要なら後段で変換してください。")
    args = p.parse_args()

    src = Path(args.db)
    if not src.exists():
        print(f"[ERR] chat.db が見つかりません: {src}", file=sys.stderr)
        return 1

    backup = Path(args.backup)
    try:
        backup_sqlite(src, backup)
    except sqlite3.OperationalError as e:
        print(f"[ERR] バックアップに失敗: {e}\n"
              f" - フルディスクアクセスをTerminal/Pythonに付与してください。\n"
              f" - Messages.appを一度終了して再実行してください。", file=sys.stderr)
        return 1

    conn = sqlite3.connect(backup)
    conn.row_factory = sqlite3.Row

    # 期間フィルタを message.date に適用
    since_ts_raw: Optional[int] = None
    if args.since_days:
        # 「今」からN日前（Apple epoch換算は後段で判定）
        now_unix = int(time.time())
        since_unix = now_unix - args.since_days * 86400
        # 後でSQLではなくPython側で判定する（ns/秒差吸収のため）
        since_ts_raw = since_unix
    elif args.since:
        try:
            dt = datetime.strptime(args.since, "%Y-%m-%d")
            since_ts_raw = int(dt.replace(tzinfo=timezone.utc).timestamp())
        except Exception:
            print(f"[WARN] --since の形式が不正です（YYYY-MM-DD）：{args.since}")

    # 付帯情報の下ごしらえ
    participants = load_participants(conn)         # chat_id -> handles[]
    attachments_map = build_attachment_map(conn)   # message_id -> [attachments...]

    # chat基本情報
    chat_rows = list(conn.execute("SELECT ROWID, guid, display_name, service_name FROM chat"))
    chat_info = {r["ROWID"]: dict(chat_id=r["ROWID"], guid=r["guid"], name=r["display_name"], service=r["service_name"]) for r in chat_rows}

    # message + 紐付け
    q_msg = """
        SELECT m.ROWID as msg_id, m.guid as msg_guid, m.text, m.attributedBody,
               m.date, m.date_read, m.date_delivered,
               m.is_from_me, m.handle_id, m.service, m.account, m.account_guid,
               cmj.chat_id
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        -- WHERE はPython側で since を判定（ns/秒問題のため）
        ORDER BY m.date ASC
    """

    msgs_flat: List[Dict[str,Any]] = []
    threads: Dict[int, Dict[str,Any]] = {}  # chat_id -> thread obj

    cur = conn.execute(q_msg)
    for r in cur:
        chat_id = r["chat_id"]
        info = chat_info.get(chat_id, {})
        raw_date = r["date"]
        raw_read = r["date_read"]
        raw_delv = r["date_delivered"]

        # since判定（UNIX秒で比較するため、normalize_date相当の補正で秒を出す）
        def to_unix_seconds(raw: Optional[int]) -> Optional[float]:
            if raw is None: return None
            v = int(raw)
            if abs(v) > 3_000_000_000:
                sec = v / 1_000_000_000
            else:
                sec = v
            return sec + APPLE_EPOCH_OFFSET

        if since_ts_raw is not None:
            unix = to_unix_seconds(raw_date)
            if unix is None or unix < since_ts_raw:
                continue

        text = r["text"]
        # 方向
        direction = "outbound" if r["is_from_me"] == 1 else "inbound"
        # 送信者
        sender_handle = None
        if r["is_from_me"] == 1:
            sender_handle = "me"
        else:
            # inbound の場合、handle_idが送信者
            if r["handle_id"] is not None:
                hrow = conn.execute("SELECT id FROM handle WHERE ROWID = ?", (r["handle_id"],)).fetchone()
                sender_handle = hrow["id"] if hrow else None

        # 添付
        atts = attachments_map.get(r["msg_id"], [])

        item = {
            "chat_id": chat_id,
            "chat_guid": info.get("guid"),
            "chat_display_name": info.get("name"),
            "service": info.get("service") or r["service"],
            "message_id": r["msg_id"],
            "message_guid": r["msg_guid"],
            "date": normalize_date(raw_date),
            "delivered_at": normalize_date(raw_delv),
            "read_at": normalize_date(raw_read),
            "direction": direction,
            "sender": sender_handle,
            "participants": participants.get(chat_id, []),
            "text": text,
            "attachments": atts,
            "account": r["account"],
            "account_guid": r["account_guid"],
        }
        msgs_flat.append(item)

        # threads へも格納
        th = threads.setdefault(chat_id, {
            "chat_id": chat_id,
            "chat_guid": info.get("guid"),
            "chat_display_name": info.get("name"),
            "service": info.get("service"),
            "participants": participants.get(chat_id, []),
            "first_date": None,
            "last_date": None,
            "message_count": 0,
            "messages": []
        })
        th["messages"].append({
            k: item[k] for k in [
                "message_id","message_guid","date","delivered_at","read_at",
                "direction","sender","text","attachments"
            ]
        })
        th["message_count"] += 1
        if item["date"]:
            th["first_date"] = th["first_date"] or item["date"]
            th["last_date"]  = item["date"]

    # 書き出し
    out_flat = Path(args.out); out_flat.parent.mkdir(parents=True, exist_ok=True)
    with out_flat.open("w", encoding="utf-8") as fp:
        for m in msgs_flat:
            fp.write(json.dumps(m, ensure_ascii=False) + "\n")

    out_threads = Path(args.threads_out); out_threads.parent.mkdir(parents=True, exist_ok=True)
    with out_threads.open("w", encoding="utf-8") as fp:
        for _, th in threads.items():
            fp.write(json.dumps(th, ensure_ascii=False) + "\n")

    print(f"wrote {len(msgs_flat)} messages -> {out_flat}")
    print(f"wrote {len(threads)} threads  -> {out_threads}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
