#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gmail (Workspace) の送信済みメールを JSONL にエクスポートするスクリプト
- サービスアカウント + ドメイン全体の権限委任（DWD）が前提
- 1行=1通で dataset/messages.jsonl に追記
- 重複防止のため dataset/.seen_ids.txt に取得済みIDを記録

使い方（例）:
  python dataset/sa_gmail_sent_to_jsonl.py \
    --sa-key-file "/path/to/service_account.json" \
    --impersonate "user@yourdomain.com" \
    --query 'in:sent newer_than:90d'

環境変数でも指定可能:
  SA_KEY_FILE, IMPERSONATE, GMAIL_QUERY, OUT_FILE, SEEN_FILE, TIMEZONE
"""
import argparse
import base64
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dateutil import tz
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm


def getenv(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Gmail Sent messages to JSONL (Service Account + DWD)",
    )
    parser.add_argument("--sa-key-file", default=getenv("SA_KEY_FILE"),
                        help="サービスアカウントのJSON鍵ファイルパス（必須/または環境変数SA_KEY_FILE）")
    parser.add_argument("--impersonate", default=getenv("IMPERSONATE"),
                        help="なりすまし対象ユーザー（必須/または環境変数IMPERSONATE）")
    parser.add_argument("--query", default=getenv("GMAIL_QUERY", "in:sent newer_than:30d"),
                        help="Gmail検索クエリ（既定: 'in:sent newer_than:30d'）")
    parser.add_argument("--label-ids", default="SENT",
                        help="カンマ区切りのラベルID（既定: SENT）")
    parser.add_argument("--max-results", type=int, default=500,
                        help="1ページの最大取得件数（<=500）")
    # 出力と管理ファイルはスクリプトのあるディレクトリ直下を既定に
    here = Path(__file__).resolve().parent
    parser.add_argument("--out", default=getenv("OUT_FILE", str(here / "messages.jsonl")),
                        help="出力JSONLパス（既定: dataset/messages.jsonl）")
    parser.add_argument("--seen-file", default=getenv("SEEN_FILE", str(here / ".seen_ids.txt")),
                        help="取得済みID記録ファイル（既定: dataset/.seen_ids.txt）")
    parser.add_argument("--timezone", default=getenv("TIMEZONE", "Asia/Tokyo"),
                        help="表示タイムゾーン（既定: Asia/Tokyo）")
    parser.add_argument("--rate-sleep", type=float, default=0.02,
                        help="各メッセージ取得のスリープ秒（既定: 0.02）")
    return parser.parse_args()


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def build_service(sa_key_file: str, impersonate: str):
    creds = service_account.Credentials.from_service_account_file(
        sa_key_file, scopes=SCOPES, subject=impersonate
    )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def header(headers: List[Dict[str, str]], name: str) -> Optional[str]:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value")
    return None


def parse_addrs(value: Optional[str]) -> List[Dict[str, Optional[str]]]:
    if not value:
        return []
    import email.utils
    res: List[Dict[str, Optional[str]]] = []
    for name, addr in email.utils.getaddresses([value]):
        res.append({"name": name or None, "email": addr})
    return res


def strip_html(s: str) -> str:
    # 簡易HTML→テキスト（タグ除去 + アンエスケープ）
    s2 = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s2)


def pick_body(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], List[Dict[str, Any]]]:
    """
    text/plain を優先し、無ければ text/html を返す。
    添付はメタ情報（filename, mime, size, attachment_id）のみ収集。
    """
    attachments: List[Dict[str, Any]] = []

    def walk(p: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        mime = p.get("mimeType")
        filename = p.get("filename")
        body = p.get("body", {})
        data = body.get("data")
        attachment_id = body.get("attachmentId")

        text_part: Optional[str] = None
        html_part: Optional[str] = None

        if filename:
            attachments.append({
                "filename": filename,
                "mime_type": mime,
                "size": body.get("size"),
                "attachment_id": attachment_id
            })

        if data:
            raw = base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", errors="ignore")
            if mime == "text/plain":
                text_part = raw
            elif mime == "text/html":
                html_part = raw

        for sp in p.get("parts", []) or []:
            t, h = walk(sp)
            text_part = text_part or t
            html_part = html_part or h

        return text_part, html_part

    t, h = walk(payload)
    return t, h, attachments


def to_iso8601_ms(ms: int, tzname: str) -> str:
    tzinfo = tz.gettz(tzname) or timezone.utc
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone(tzinfo)
    return dt.isoformat(timespec="seconds")


def sha256(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_seen(path: Path) -> set:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return set(x.strip() for x in f if x.strip())


def append_seen(path: Path, ids: List[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for i in ids:
            f.write(i + "\n")


def main() -> int:
    args = parse_args()

    if not args.sa_key_file or not Path(args.sa_key_file).exists():
        print("ERROR: --sa-key-file が未指定か存在しません。", file=sys.stderr)
        return 2
    if not args.impersonate:
        print("ERROR: --impersonate が未指定です。", file=sys.stderr)
        return 2

    label_ids = [x.strip() for x in (args.label_ids.split(",") if args.label_ids else []) if x.strip()]
    if not label_ids:
        label_ids = ["SENT"]

    out_path = Path(args.out).resolve()
    seen_path = Path(args.seen_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        service = build_service(args.sa_key_file, args.impersonate)
        # 簡単な疎通確認（権限/なりすましチェック）
        _ = service.users().getProfile(userId="me").execute()
    except HttpError as e:
        print(f"ERROR: Gmail API 呼び出しエラー: {e}", file=sys.stderr)
        return 1

    seen = load_seen(seen_path)
    new_ids: List[str] = []
    total = 0
    page_token: Optional[str] = None

    with out_path.open("a", encoding="utf-8") as out_fp:
        while True:
            try:
                resp = service.users().messages().list(
                    userId="me",
                    labelIds=label_ids,
                    q=args.query,
                    pageToken=page_token,
                    includeSpamTrash=False,
                    maxResults=min(max(args.max_results, 1), 500)
                ).execute()
            except HttpError as e:
                print(f"ERROR: list 失敗: {e}", file=sys.stderr)
                break

            msgs = resp.get("messages", [])
            ids = [m["id"] for m in msgs if m.get("id") not in seen]

            if not ids and not resp.get("nextPageToken"):
                break

            for mid in tqdm(ids, desc="fetching", unit="msg"):
                try:
                    m = service.users().messages().get(
                        userId="me", id=mid, format="full"
                    ).execute()
                except HttpError as e:
                    print(f"WARN: get失敗 id={mid}: {e}", file=sys.stderr)
                    continue

                headers = m.get("payload", {}).get("headers", [])
                text, html_body, atts = pick_body(m.get("payload", {}))

                # 正規化本文（text優先、無ければhtml簡易テキスト化）
                normalized = (text or (strip_html(html_body) if html_body else "")).strip()

                item: Dict[str, Any] = {
                    "id": m.get("id"),
                    "date": to_iso8601_ms(int(m.get("internalDate")), args.timezone) if m.get("internalDate") else None,
                    "from": (parse_addrs(header(headers, "From")) or [None])[0],
                    "to": parse_addrs(header(headers, "To")),
                    "cc": parse_addrs(header(headers, "Cc")),
                    "bcc": parse_addrs(header(headers, "Bcc")),
                    "subject": header(headers, "Subject"),
                    "body_text": text,
                    "body_html": (html_body if html_body and not text else None),
                    "attachments": atts,
                    "thread_id": m.get("threadId"),
                    "in_reply_to": header(headers, "In-Reply-To"),
                    "references": [x.strip() for x in (header(headers, "References") or "").split()] or [],
                    "labels": m.get("labelIds", []),
                    "language": "ja",  # 必要に応じて自動判定へ差し替え
                    "source": {
                        "provider": "gmail",
                        "message_id": header(headers, "Message-Id"),
                        "sync_time": datetime.now(tz.gettz(args.timezone) or timezone.utc).isoformat(timespec="seconds"),
                    },
                    "hash": sha256(normalized) if normalized else None
                }

                out_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
                total += 1
                new_ids.append(mid)
                time.sleep(max(args.rate_sleep, 0.0))

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    if new_ids:
        append_seen(seen_path, new_ids)

    print(f"wrote {total} messages -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
