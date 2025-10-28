#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gmail 送信済み → JSONL（OAuth 版）
- HTML→テキストのフォールバックを強化
- 署名/フッター/引用の簡易除去オプション（既定ON）
- 全期間取得オプション: --all（= クエリを外して SENT 全件）
"""
import argparse, base64, hashlib, html, json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dateutil import tz
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Export Gmail Sent messages to JSONL (OAuth)")
    p.add_argument("--client-secret", default=os.getenv("CLIENT_SECRET_FILE","client_secret.json"))
    p.add_argument("--token-file", default=os.getenv("TOKEN_FILE", str(here / "token.json")))
    p.add_argument("--query", default=os.getenv("GMAIL_QUERY","in:sent newer_than:30d"),
                   help="Gmail 検索クエリ（例: 'in:sent after:2024/01/01'）")
    p.add_argument("--label-ids", default="SENT", help="カンマ区切り。既定: SENT")
    p.add_argument("--out", default=os.getenv("OUT_FILE", str(here / "messages.jsonl")))
    p.add_argument("--seen-file", default=os.getenv("SEEN_FILE", str(here / ".seen_ids.txt")))
    p.add_argument("--timezone", default=os.getenv("TIMEZONE","Asia/Tokyo"))
    p.add_argument("--rate-sleep", type=float, default=0.02)
    # 新規: 全期間取得
    p.add_argument("--all", dest="all_time", action="store_true",
                   help="全期間取得（クエリ無指定で SENT 全件を対象）")
    # 新規: HTML→テキストと整形の挙動
    p.add_argument("--no-strip-footer", action="store_true",
                   help="署名/フッター/引用の除去を無効化（既定は除去ON）")
    return p.parse_args()

def header(headers: List[Dict[str,str]], name:str)->Optional[str]:
    for h in headers:
        if h.get("name","").lower()==name.lower():
            return h.get("value")
    return None

def parse_addrs(v: Optional[str]) -> List[Dict[str,Optional[str]]]:
    if not v: return []
    import email.utils
    return [{"name": n or None, "email": a} for n,a in email.utils.getaddresses([v])]

# --- HTML→Text 強化 ---
_BR_RX = re.compile(r"(?i)<\s*br\s*/?>")
_BLOCK_RX = re.compile(r"(?i)</\s*(p|div|li|h[1-6]|tr)\s*>")
_TAG_RX = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RX = re.compile(r"(?is)<\s*(script|style)[^>]*>.*?</\s*\1\s*>")
WS_RX = re.compile(r"[ \t\u3000]+")
BLANKS_RX = re.compile(r"\n{3,}")

def html_to_text(s: str) -> str:
    # 改行を保つタグを先に改行化
    s = SCRIPT_STYLE_RX.sub("", s)
    s = _BR_RX.sub("\n", s)
    s = _BLOCK_RX.sub("\n", s)
    s = _TAG_RX.sub("", s)
    s = html.unescape(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = WS_RX.sub(" ", s)
    s = BLANKS_RX.sub("\n\n", s)
    return s.strip()

# --- 署名/引用/フッターの簡易除去 ---
FOOTER_PATTERNS = [
    r"^--\s*$",                                 # 標準署名区切り
    r"^Sent from my iPhone",                    # モバイル署名
    r"^-{5,}\s*Original Message\s*-{5,}$",      # -----Original Message-----
    r"^>+ .*$",                                 # 引用行（頭が '>'）
    r"^On .+ wrote:$",                          # On ... wrote:
    r"^From:\s.*$",                             # メール引用ヘッダ
    r"^このメールは.*自動.*送信",                  # 日本語フッター例
    r"^This email.*confidential",               # 英語ディスクレーマ
]
FOOTER_RX = re.compile("|".join(f"({p})" for p in FOOTER_PATTERNS), re.IGNORECASE | re.MULTILINE)

def strip_footer_and_quotes(text: str) -> str:
    # 区切りが現れた地点以降を落とす（最初のヒットを使用）
    m = FOOTER_RX.search(text)
    if m:
        text = text[:m.start()]
    # 連続する引用行の塊を削る（行頭 '>' を持つ段落）
    lines = text.split("\n")
    kept = []
    for ln in lines:
        if ln.lstrip().startswith(">"):
            continue
        kept.append(ln)
    text = "\n".join(kept)
    # 体裁整え
    text = BLANKS_RX.sub("\n\n", text).strip()
    return text

def strip_html_simple(s: str) -> str:
    # 後方互換（旧関数名）
    return html_to_text(s)

def pick_body(payload: Dict[str,Any]):
    atts=[]
    def walk(p):
        mime=p.get("mimeType"); fn=p.get("filename"); body=p.get("body",{})
        data=body.get("data"); att_id=body.get("attachmentId")
        text=None; htmlp=None
        if fn:
            atts.append({"filename":fn,"mime_type":mime,"size":body.get("size"),"attachment_id":att_id})
        if data:
            raw=base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8","ignore")
            if mime=="text/plain": text=raw
            elif mime=="text/html": htmlp=raw
        for sp in p.get("parts",[]) or []:
            t,h=walk(sp); text=text or t; htmlp=htmlp or h
        return text, htmlp
    t,h=walk(payload)
    return t,h,atts

def to_iso8601_ms(ms: int, tzname: str) -> str:
    tzinfo = tz.gettz(tzname) or timezone.utc
    dt = datetime.fromtimestamp(ms/1000, tz=timezone.utc).astimezone(tzinfo)
    return dt.isoformat(timespec="seconds")

def sha256(s:str)->str:
    return "sha256:"+hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_seen(p: Path)->set:
    if not p.exists(): return set()
    return set(x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip())

def append_seen(p: Path, ids: List[str]):
    with p.open("a",encoding="utf-8") as f:
        for i in ids: f.write(i+"\n")

def get_creds(client_secret: str, token_file: str):
    creds=None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secret, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file,"w") as f:
            f.write(creds.to_json())
    return creds

def main():
    args = parse_args()
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_path = Path(args.seen_file); seen_path.parent.mkdir(parents=True, exist_ok=True)

    creds = get_creds(args.client_secret, args.token_file)
    service = build("gmail","v1",credentials=creds, cache_discovery=False)

    try:
        _ = service.users().getProfile(userId="me").execute()
    except HttpError as e:
        print(f"ERROR profile: {e}", file=sys.stderr); return 1

    label_ids = [x.strip() for x in (args.label_ids.split(",") if args.label_ids else []) if x.strip()] or ["SENT"]
    seen = load_seen(seen_path)
    page=None; total=0; new=[]

    with out_path.open("a",encoding="utf-8") as out_fp:
        while True:
            try:
                kwargs = dict(userId="me", labelIds=label_ids, includeSpamTrash=False, maxResults=500, pageToken=page)
                # 全期間取得 --all の時は q を外す（=全件）
                if not args.all_time and args.query:
                    kwargs["q"] = args.query
                resp = service.users().messages().list(**kwargs).execute()
            except HttpError as e:
                print(f"ERROR list: {e}", file=sys.stderr); break

            msgs = resp.get("messages",[])
            ids = [m["id"] for m in msgs if m["id"] not in seen]
            if not ids and not resp.get("nextPageToken"): break

            for mid in ids:
                try:
                    m = service.users().messages().get(userId="me", id=mid, format="full").execute()
                except HttpError as e:
                    print(f"WARN get {mid}: {e}", file=sys.stderr); continue

                headers = m.get("payload",{}).get("headers",[])
                text_raw, html_raw, atts = pick_body(m.get("payload",{}))

                # 本文の最終決定ロジック：
                # 1) text/plain があればそれを基準
                # 2) なければ HTML を強化コンバータでテキスト化
                text = (text_raw or "")
                if not text and html_raw:
                    text = html_to_text(html_raw)

                # 署名/引用の簡易除去（既定ON）
                if not args.no_strip_footer and text:
                    text = strip_footer_and_quotes(text)

                normalized = (text or (html_to_text(html_raw) if html_raw else "")).strip()

                item = {
                    "id": m.get("id"),
                    "date": to_iso8601_ms(int(m.get("internalDate")), args.timezone) if m.get("internalDate") else None,
                    "from": (parse_addrs(header(headers,"From")) or [None])[0],
                    "to": parse_addrs(header(headers,"To")),
                    "cc": parse_addrs(header(headers,"Cc")),
                    "bcc": parse_addrs(header(headers,"Bcc")),
                    "subject": header(headers,"Subject"),
                    "body_text": text or None,
                    # text がある場合は body_html は省略（冗長回避）
                    "body_html": (html_raw if html_raw and not text_raw else None),
                    "attachments": atts,
                    "thread_id": m.get("threadId"),
                    "in_reply_to": header(headers,"In-Reply-To"),
                    "references": [x.strip() for x in (header(headers,"References") or "").split()] or [],
                    "labels": m.get("labelIds",[]),
                    "language": "ja",
                    "source": {
                        "provider": "gmail",
                        "message_id": header(headers,"Message-Id"),
                        "sync_time": datetime.now(tz.gettz(args.timezone) or timezone.utc).isoformat(timespec="seconds"),
                    },
                    "hash": sha256(normalized) if normalized else None
                }
                out_fp.write(json.dumps(item, ensure_ascii=False)+"\n")
                total += 1; new.append(mid)
                time.sleep(max(args.rate_sleep,0.0))

            page = resp.get("nextPageToken")
            if not page: break

    if new: append_seen(seen_path, new)
    print(f"wrote {total} messages -> {out_path}")
    return 0

if __name__=="__main__":
    sys.exit(main())
