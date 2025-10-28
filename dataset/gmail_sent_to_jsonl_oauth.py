#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    p.add_argument("--query", default=os.getenv("GMAIL_QUERY","in:sent newer_than:30d"))
    p.add_argument("--label-ids", default="SENT")
    p.add_argument("--out", default=os.getenv("OUT_FILE", str(here / "messages.jsonl")))
    p.add_argument("--seen-file", default=os.getenv("SEEN_FILE", str(here / ".seen_ids.txt")))
    p.add_argument("--timezone", default=os.getenv("TIMEZONE","Asia/Tokyo"))
    p.add_argument("--rate-sleep", type=float, default=0.02)
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

def strip_html(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>","", s))

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
                resp = service.users().messages().list(
                    userId="me", labelIds=label_ids, q=args.query,
                    pageToken=page, includeSpamTrash=False, maxResults=500
                ).execute()
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
                text, htmlp, atts = pick_body(m.get("payload",{}))
                normalized = (text or (strip_html(htmlp) if htmlp else "")).strip()
                item = {
                    "id": m.get("id"),
                    "date": to_iso8601_ms(int(m.get("internalDate")), args.timezone) if m.get("internalDate") else None,
                    "from": (parse_addrs(header(headers,"From")) or [None])[0],
                    "to": parse_addrs(header(headers,"To")),
                    "cc": parse_addrs(header(headers,"Cc")),
                    "bcc": parse_addrs(header(headers,"Bcc")),
                    "subject": header(headers,"Subject"),
                    "body_text": text,
                    "body_html": (htmlp if htmlp and not text else None),
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
