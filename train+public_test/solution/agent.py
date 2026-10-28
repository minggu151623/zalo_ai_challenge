"""LifeMate agent: prompt building, tool-call parsing and a batched agent loop over zaic_tools.

The LLM backend is any function `generate(list_of_message_lists, list_of_tool_lists) -> list[str]` returning raw
assistant text in Qwen/Hermes format (<tool_call>{"name": ..., "arguments": {...}}</tool_call>).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ nhật"]
MAX_STEPS = 8
MAX_RETRIES = 2
HISTORY_RESULT_CHARS = 1200
RESULT_CHARS = 4000

SYSTEM_RULES = """Bạn là trợ lý giọng nói LifeMate. Bạn hành động bằng cách gọi tool của các app đã cài trên thiết bị, được chấm theo đúng các tool call và trạng thái app sau cùng.

Quy tắc:
1. Chỉ gọi tool có trong danh sách. Nếu không app nào đã cài làm được yêu cầu, hoặc người dùng tự huỷ yêu cầu ("à thôi khỏi"), hoặc chỉ trò chuyện/cảm ơn, thì KHÔNG gọi tool nào, chỉ trả lời ngắn.
2. Tham số là ID (device_id, contact_id, event_id, bill_id, ...) phải lấy từ kết quả tool đọc tương ứng (get_status, list_..., contacts.search ...). Nếu ID đã có trong kết quả ở các lượt trước thì dùng lại, không gọi đọc lại.
3. Tìm người trong danh bạ: contacts.search với query đúng tên/quan hệ người dùng nói (ví dụ "Sếp Dũng", "mẹ").
4. Nếu thiếu tham số bắt buộc mà không suy ra được từ lời nói, ngữ cảnh hay dữ liệu (ví dụ tìm thấy nhiều người cùng tên, không rõ thiết bị nào, không nói số tiền/giờ), gọi sys.ask_missing_info với tool_name là tool định gọi và params là tên các tham số còn thiếu, rồi dừng. Các phần khác của yêu cầu đã đủ thông tin thì vẫn làm.
5. Yêu cầu có điều kiện ("nếu mai mưa thì ..."): kiểm tra điều kiện bằng tool đọc (hoặc kết quả đã có ở lượt trước); điều kiện không thoả thì không làm hành động đó.
6. Không gọi tool thừa: chỉ đọc khi cần dữ liệu để làm hoặc trả lời yêu cầu hiện tại. Không gọi lại một tool đã thành công với cùng tham số.
7. Thời gian dạng ISO 8601 có múi giờ, ví dụ 2026-11-29T09:00:00+07:00; ngày dạng YYYY-MM-DD. Số nguyên không có dấu chấm/phẩy. Chỉ truyền tham số người dùng yêu cầu hoặc tool bắt buộc.
8. Văn bản tự do (nội dung tin nhắn, tiêu đề, ghi chú) viết ngắn gọn bằng tiếng Việt, sát ý người dùng.
9. Khi đã xong, trả lời một câu ngắn bằng tiếng Việt, không kèm tool call."""


def _clip(obj, n):
    s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return s if len(s) <= n else s[:n] + "…"


def date_table(now: datetime, days=14):
    rows = []
    for i in range(-1, days):
        d = now + timedelta(days=i)
        tag = {-1: " (hôm qua)", 0: " (hôm nay)", 1: " (ngày mai)", 2: " (ngày kia)"}.get(i, "")
        rows.append(f"{WEEKDAYS[d.weekday()]} {d:%Y-%m-%d}{tag}")
    return "; ".join(rows)


def system_prompt(item):
    ctx = item["context"]
    now = datetime.fromisoformat(ctx["current_time"])
    prof = ctx.get("user_profile") or {}
    loc = ctx.get("location") or {}
    lines = [
        SYSTEM_RULES,
        "",
        "Ngữ cảnh:",
        f"- Bây giờ: {WEEKDAYS[now.weekday()]}, {ctx['current_time']} ({ctx.get('timezone', '')})",
        f"- Lịch: {date_table(now)}",
        f"- Vị trí hiện tại: lat {loc.get('lat')}, lng {loc.get('lng')}",
        "- Hồ sơ người dùng: " + json.dumps(prof, ensure_ascii=False),
        "- App đã cài: " + ", ".join(ctx["installed_apps"]),
    ]
    return "\n".join(lines)


def _desc(t):
    d = t.get("description", "")
    kind = {"read": "[đọc] ", "write": "[ghi] ", "meta": ""}.get(t.get("kind"), "")
    if t.get("returns"):
        d += f" → trả về {t['returns']}"
    return kind + d


def _params(p):
    props = {}
    for k, v in p.get("properties", {}).items():
        v = dict(v)
        ref = v.pop("x-ref", None)
        v.pop("x-match", None)
        if ref and "description" not in v:
            v["description"] = f"ID trong {ref}"
        props[k] = v
    out = {"type": "object", "properties": props, "required": p.get("required", [])}
    if p.get("x-one-of"):
        out["description"] = "Bắt buộc một trong các nhóm: " + " | ".join("+".join(g) for g in p["x-one-of"])
    return out


def build_tools(item, catalog_tools):
    names = [t for a in item["context"]["installed_apps"].values() for t in a["tools"]] + ["sys.ask_missing_info"]
    tools = []
    for n in dict.fromkeys(names):
        t = catalog_tools.get(n)
        if t:
            tools.append({"type": "function", "function": {"name": n, "description": _desc(t), "parameters": _params(t["parameters"])}})
    return tools


def _tc(name, args):
    return {"type": "function", "function": {"name": name, "arguments": args}}


def history_messages(item, user_texts):
    """user_texts: list with the transcript of every user turn in history, in order."""
    msgs, ui = [], 0
    for h in item.get("history") or []:
        if h["role"] == "user":
            msgs.append({"role": "user", "content": user_texts[ui]})
            ui += 1
            continue
        calls = h.get("tool_calls") or []
        results = h.get("tool_results") or []
        if calls:
            msgs.append({"role": "assistant", "content": "", "tool_calls": [_tc(c["name"], c.get("arguments") or {}) for c in calls]})
            for i, c in enumerate(calls):
                r = results[i] if i < len(results) else {"ok": True}
                msgs.append({"role": "tool", "content": _clip(r, HISTORY_RESULT_CHARS)})
        msgs.append({"role": "assistant", "content": h.get("text") or ""})
    return msgs


_TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def parse_tool_calls(text):
    calls, bad = [], []
    for raw in _TC_RE.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            bad.append(raw)
            continue
        args = obj.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(obj.get("name"), str):
            calls.append((obj["name"], args if isinstance(args, dict) else {}))
    return calls, bad


class AgentState:
    def __init__(self, item, env, tools, user_texts, final_text):
        self.item, self.env, self.tools = item, env, tools
        self.messages = [{"role": "system", "content": system_prompt(item)}] + history_messages(item, user_texts) + \
                        [{"role": "user", "content": final_text}]
        self.done = False
        self.steps = 0
        self.retries = 0
        self.done_calls = set()

    def step(self, text):
        """Consume one assistant output; execute its tool calls; decide whether the episode is over."""
        self.steps += 1
        calls, bad = parse_tool_calls(text)
        content = text.split("<tool_call>")[0].strip()
        if not calls:
            if bad and self.retries < MAX_RETRIES:
                self.retries += 1
                self.messages.append({"role": "assistant", "content": text})
                self.messages.append({"role": "user", "content": "Tool call không phải JSON hợp lệ, hãy gọi lại đúng định dạng."})
            else:
                self.messages.append({"role": "assistant", "content": content})
                self.done = True
            return
        self.messages.append({"role": "assistant", "content": content, "tool_calls": [_tc(n, a) for n, a in calls]})
        stop = False
        for name, args in calls:
            key = name + json.dumps(args, sort_keys=True, ensure_ascii=False)
            if key in self.done_calls:
                res = {"ok": True, "note": "đã gọi tool này với cùng tham số, dùng kết quả trước"}
            else:
                # validate first: a rejected call would be logged and count as unmatched in the score
                v = self.env.validate(name, args)
                if not v["ok"]:
                    res = {"ok": False, "error": v["error"], "note": "chưa thực thi, hãy sửa lại tham số"}
                    self.retries += 1
                else:
                    res = self.env.call(name, args)
                    if self.env.kind(name) == "read":  # repeated writes can be intended (redeem twice)
                        self.done_calls.add(key)
            if name == "sys.ask_missing_info" and res.get("ok"):
                stop = True
            self.messages.append({"role": "tool", "content": _clip(res, RESULT_CHARS)})
        if stop or self.steps >= MAX_STEPS or self.retries > MAX_RETRIES + 2:
            self.done = True


def run_agents(states, generate, log=print, on_done=None, wave=0):
    """Batched loop: every round sends unfinished episodes to the backend together.
    wave > 0 runs episodes in waves of that size so finished ones are reported (on_done) early."""
    waves = [states[i:i + wave] for i in range(0, len(states), wave)] if wave else [states]
    for wi, group in enumerate(waves, 1):
        rnd = 0
        while True:
            live = [s for s in group if not s.done]
            if not live:
                break
            rnd += 1
            outs = generate([s.messages for s in live], [s.tools for s in live])
            for s, o in zip(live, outs):
                s.step(o)
                if s.done and on_done:
                    on_done(s)
            log(f"wave {wi}/{len(waves)} round {rnd}: {len(live)} live -> {sum(not s.done for s in group)} remaining")
