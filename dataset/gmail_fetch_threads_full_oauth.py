#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
送信済みが含まれるスレッドを“返信も含めて”丸ごと取得（OAuth）
- --all で全期間、--query で範囲指定
- 各メッセージに direction: outbound/inbound を付与
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

# ---- HTML→Text 強化 ----
_BR_RX = re.compile(r"(?i)<\s*br\s*/?>")
_BLOCK_RX = re.compile(r"(?i)</\s*(p|div|li|h[1-6]|tr)\s*>")
_TAG_RX = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RX = re.compile(r"(?is)<\s*(script|style)[^>]*>.*?</\s*\1\s*>")
WS_RX = re.compile(r"[ \t\u3000]+")
BLANKS_RX = re.compile(r"\n{3,}")
FOOTER_PATTERNS = [
    r"^--\s*$",
    r"^Sent from my iPhone",
    r"^-{5,}\s*Original Message\s*-{5,}$",
    r"^>+ .*$",
    r"^On .+ wrote:$",
    r"^From:\s.*$",
    r"^このメールは.*自動.*送信",
    r"^This email.*confidential",
]
FOOTER_RX = re.compile("|".join(f"({p})" for p in FOOTER_PATTERNS), re.IGNORECASE | re.MULTILINE)

def html_to_text(s: str) -> str:
    s = SCRIPT_STYLE_RX.sub("", s)
    s = _BR_RX.sub("\n", s)
    s = _BLOCK_RX.sub("\n", s)
    s = _TAG_RX.sub("", s)
    s = html.unescape(s)
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = WS_RX.sub(" ", s)
    s = BLANKS_RX.sub("\n\n", s)
    return s.strip()

def strip_footer_and_quotes(text: str) -> str:
    m = FOOTER_RX.search(text)
    if m:
        text = text[:m.start()]
    lines = text.split("\n")
    kept = [ln for ln in lines if not ln.lstrip().startswith(">")]
    text = "\n".join(kept)
    text = BLANKS_RX.sub("\n\n", text).strip()
    return text

# ---- ヘルパ ----
def header(headers: List[Dict[str,str]], name:str)->Optional[str]:
    for h in headers:
        if h.get("name","").lower()==name.lower():
            return h.get("value")
    return None

def parse_addrs(v: Optional[str]) -> List[Dict[str,Optional[str]]]:
    if not v: return []
    import email.utils
    return [{"name": n or None, "email": a} for n,a in email.utils.getaddresses([v])]

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
    import hashlib
    return "sha256:"+hashlib.sha256(s.encode("utf-8")).hexdigest()

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

def parse_args():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Fetch full Gmail threads that contain your SENT messages (OAuth)")
    p.add_argument("--client-secret", default=os.getenv("CLIENT_SECRET_FILE","client_secret.json"))
    p.add_argument("--token-file", default=os.getenv("TOKEN_FILE", str(here / "token.json")))
    p.add_argument("--query", default=os.getenv("GMAIL_QUERY","in:sent newer_than:30d"),
                   help="まずSENTメッセージを抽出するクエリ。例: 'in:sent after:2024/01/01'")
    p.add_argument("--all", dest="all_time", action="store_true",
                   help="全期間のSENTを対象（=クエリを外す）")
    p.add_argument("--label-ids", default="SENT", help="最初の抽出に使うラベル（既定:SENT）")
    p.add_argument("--out-threads", default=str(here / "threads_full.jsonl"),
                   help="出力ファイル（1行=1スレッド）")
    p.add_argument("--messages-out", default=None,
                   help="全メッセージを平坦化して出力するJSONL（任意）")
    p.add_argument("--timezone", default=os.getenv("TIMEZONE","Asia/Tokyo"))
    p.add_argument("--rate-sleep", type=float, default=0.01)
    p.add_argument("--no-strip-footer", action="store_true",
                   help="署名/引用の簡易除去を無効化")
    return p.parse_args()

def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    out_threads = Path(args.out_threads); out_threads.parent.mkdir(parents=True, exist_ok=True)
    out_msgs = Path(args.messages_out) if args.messages_out else None
    if out_msgs: out_msgs.parent.mkdir(parents=True, exist_ok=True)

    creds = get_creds(args.client_secret, args.token_file)
    svc = build("gmail","v1",credentials=creds, cache_discovery=False)

    prof = svc.users().getProfile(userId="me").execute()
    my_email = (prof.get("emailAddress") or "").lower()

    # 1) 送信済みメッセージ→threadId を収集
    label_ids = [x.strip() for x in (args.label_ids.split(",") if args.label_ids else []) if x.strip()] or ["SENT"]
    thread_ids = []
    page=None
    while True:
        kwargs = dict(userId="me", labelIds=label_ids, includeSpamTrash=False, maxResults=500, pageToken=page)
        if not args.all_time and args.query:
            kwargs["q"] = args.query
        resp = svc.users().messages().list(**kwargs).execute()
        msgs = resp.get("messages",[])
        for m in msgs:
            tid = m.get("threadId")
            if tid:
                thread_ids.append(tid)
        page = resp.get("nextPageToken")
        if not page:
            break
    thread_ids = list(dict.fromkeys(thread_ids))  # uniq

    # 2) 各スレッドを展開
    th_fp = out_threads.open("w", encoding="utf-8")
    msg_fp = out_msgs.open("w", encoding="utf-8") if out_msgs else None
    tzname = args.timezone

    for tid in thread_ids:
        try:
            th = svc.users().threads().get(userId="me", id=tid, format="full").execute()
        except HttpError as e:
            print(f"WARN threads.get {tid}: {e}", file=sys.stderr)
            continue

        participants=set()
        labels_union=set()
        msgs_out=[]
        subj=None
        first_date=None
        last_date=None

        for m in th.get("messages", []):
            headers = m.get("payload",{}).get("headers",[])
            date_ms = int(m.get("internalDate")) if m.get("internalDate") else None
            date_iso = to_iso8601_ms(date_ms, tzname) if date_ms else None
            fr = (parse_addrs(header(headers,"From")) or [None])[0]
            tos = parse_addrs(header(headers,"To"))
            ccs = parse_addrs(header(headers,"Cc"))
            bccs= parse_addrs(header(headers,"Bcc"))
            if not subj:
                subj = header(headers,"Subject") or subj

            for v in [fr] + tos + ccs + bccs:
                if v and v.get("email"):
                    participants.add(v["email"].lower())

            t_raw, h_raw, atts = pick_body(m.get("payload",{}))
            text = (t_raw or "")
            if not text and h_raw:
                text = html_to_text(h_raw)
            if not args.no_strip_footer and text:
                text = strip_footer_and_quotes(text)
            normalized = (text or (html_to_text(h_raw) if h_raw else "")).strip()

            direction = "outbound" if (fr and (fr.get("email") or "").lower()==my_email) else "inbound"

            obj = {
                "id": m.get("id"),
                "thread_id": tid,
                "date": date_iso,
                "direction": direction,
                "from": fr,
                "to": tos,
                "cc": ccs,
                "bcc": bccs,
                "subject": header(headers,"Subject"),
                "body_text": text or None,
                "body_html": (h_raw if h_raw and not t_raw else None),
                "attachments": atts,
                "labels": m.get("labelIds",[]),
                "source": {
                    "provider": "gmail",
                    "message_id": header(headers,"Message-Id"),
                },
                "hash": sha256(normalized) if normalized else None
            }
            msgs_out.append(obj)
            labels_union.update(obj["labels"])
            if date_iso:
                first_date = first_date or date_iso
                last_date = date_iso

            if msg_fp:
                msg_fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

            time.sleep(max(args.rate_sleep, 0.0))  # ← 修正済み

        thread_obj = {
            "thread_id": tid,
            "participants": sorted(participants),
            "subject": subj,
            "first_date": first_date,
            "last_date": last_date,
            "message_count": len(msgs_out),
            "labels_union": sorted(labels_union),
            "messages": msgs_out,
        }
        th_fp.write(json.dumps(thread_obj, ensure_ascii=False) + "\n")

    th_fp.close()
    if msg_fp: msg_fp.close()
    print(f"wrote {len(thread_ids)} threads -> {out_threads}")
    if out_msgs:
        print(f"flattened messages -> {out_msgs}")

if __name__=="__main__":
    sys.exit(main())
