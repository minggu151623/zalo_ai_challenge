"""Synthetic train-format samples for apps that appear only in public_test (their schemas come from the public catalog;
states and requests are invented here, nothing is read from public_test samples or environments).

    python solution/synth.py --n 600 --out out/synth.jsonl
Each sample borrows the context and seen-app state of a random train sample (contacts, sms, reminders...), adds one
unseen app with a generated state, and gets gold calls written by a per-app task template. The calls are executed with
zaic_tools' Simulator, so tool_results are real. Output lines look like train.jsonl ({id, audio_text, context, history,
label: {tool_calls, tool_results}}) plus "states" (initial state of every installed app) so sft_data.py can use them.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "public_test"))
from zaic_tools.executor import Simulator  # noqa: E402

WD = ["thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy", "chủ nhật"]
NUM = {1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười"}
TAIL = ["nhé", "nha", "giúp mình", "giùm tôi", "nhé em", "đi em", "với", "hen", "nghen", ""]
HEAD = ["", "", "Em ơi, ", "Ờ, ", "Trợ lý ơi, ", "À, ", "Này, ", "Ê em, "]


class Ctx:
    """One sample under construction: random source, current time, and helpers for spoken numbers and dates."""

    def __init__(self, rng, now):
        self.r, self.now = rng, now

    def pick(self, xs):
        return self.r.choice(xs)

    def say(self, core):
        s = self.pick(HEAD) + core
        t = self.pick(TAIL)
        s = (s + (" " + t if t else "")).strip()
        return s[0].upper() + s[1:] + self.pick([".", ".", "!", ""])

    def n(self, k):
        """Small count as a word or a digit, the way people (and ASR) write it."""
        return NUM[k] if k in NUM and self.r.random() < 0.6 else str(k)

    def money(self, v):
        if v % 1_000_000 == 0:
            return f"{v // 1_000_000} triệu"
        if v % 1000 == 0:
            k = v // 1000
            return self.pick([f"{k} nghìn", f"{k} ngàn", f"{k}k"]) if k < 1000 else f"{k // 1000} triệu {k % 1000}"
        return str(v)

    def day(self, offset):
        d = (self.now + timedelta(days=offset)).date()
        word = {0: "hôm nay", 1: "mai", 2: "ngày kia", -1: "hôm qua"}.get(offset)
        if word is None:
            weeks = (d - self.now.date()).days // 7 + (d.weekday() < self.now.weekday())  # 0: same Mon-Sun week
            wd = f"{WD[d.weekday()]} này" if weeks == 0 else f"{WD[d.weekday()]} tuần sau" if weeks == 1 else None
            word = self.pick([w for w in (wd, f"ngày {d.day} tháng {d.month}") if w])
        return d.isoformat(), word

    def at(self, offset, hour, minute=0):
        d = (self.now + timedelta(days=offset)).replace(hour=hour, minute=minute, second=0, microsecond=0)
        return d.isoformat()


def hour_words(c, h, m=0):
    part = "sáng" if h < 11 else "trưa" if h < 13 else "chiều" if h < 18 else "tối"
    hh = h if h <= 12 else h - 12
    mm = "" if m == 0 else " rưỡi" if m == 30 else f" {m}"
    return f"{hh} giờ{mm} {part}" if c.r.random() < 0.7 else f"{h}h{m:02d}" if m else f"{h} giờ"


# ---------------------------------------------------------------------------------------------- app generators
# Each returns (state, [task, ...]); a task is (utterance, [planned calls]) where a planned call is (name, args) or a
# callable(results_so_far) -> (name, args) | None for steps that depend on earlier results (conditions).

def gen_stocks(c):
    pool = [("FPT", "FPT", 128000), ("VNM", "Vinamilk", 61500), ("VCB", "Vietcombank", 89200), ("HPG", "Hòa Phát", 26750),
            ("MWG", "Thế Giới Di Động", 58300), ("TCB", "Techcombank", 23400), ("MSN", "Masan", 74100), ("SSI", "SSI", 31850),
            ("ACB", "ACB", 24900), ("VIC", "Vingroup", 42600)]
    c.r.shuffle(pool)
    held = [{"symbol": s, "name": n, "quantity": c.pick([100, 200, 300, 500, 1000]), "avg_price": p - c.r.randint(-5, 12) * 500,
             "price": p, "change_pct": round(c.r.uniform(-3, 3), 2)} for s, n, p in pool[:3]]
    quotes = [{"symbol": s, "name": n, "price": p, "change_pct": round(c.r.uniform(-4, 4), 2)} for s, n, p in pool[3:8]]
    st = {"holdings": held, "orders": [], "alerts": [], "quotes": quotes}
    hs, hn, hp = held[0]["symbol"], held[0]["name"], held[0]["price"]
    qs, qn, qp = quotes[0]["symbol"], quotes[0]["name"], quotes[0]["price"]
    k = c.pick([100, 200, 500, 1000])
    lim = (qp // 1000 - c.r.randint(1, 3)) * 1000
    thr = (qp // 1000 + c.r.randint(2, 8)) * 1000
    cond_lo = (qp // 1000 + 2) * 1000  # condition true: price below this
    tasks = [
        (c.say(f"giá cổ phiếu {qs} {c.pick(['hôm nay', 'bây giờ', 'đang'])} bao nhiêu"), [("stocks.get_quote", {"symbol": qs})]),
        (c.say(f"xem giá mã {qn if qn != qs else qs} giúp tôi"), [("stocks.get_quote", {"symbol": qs})]),
        (c.say(c.pick(["danh mục chứng khoán của tôi đang thế nào", "tôi đang giữ những mã nào", "tổng giá trị cổ phiếu tôi đang có là bao nhiêu"])),
         [("stocks.get_portfolio", {})]),
        (c.say(f"mua {k} cổ phiếu {qs}"), [("stocks.place_order", {"symbol": qs, "side": "buy", "quantity": k})]),
        (c.say(f"đặt lệnh mua {k} cổ {qs} giá {c.money(lim)}"), [("stocks.place_order", {"symbol": qs, "side": "buy", "quantity": k, "price": lim})]),
        (c.say(f"bán {held[0]['quantity'] // 2} cổ phiếu {hs}"), [("stocks.place_order", {"symbol": hs, "side": "sell", "quantity": held[0]["quantity"] // 2})]),
        (c.say(f"bán hết số cổ phiếu {hs} tôi đang có"), [("stocks.get_portfolio", {}), ("stocks.place_order", {"symbol": hs, "side": "sell", "quantity": held[0]["quantity"]})]),
        (c.say(f"báo tôi khi {qs} {c.pick(['lên trên', 'vượt'])} {c.money(thr)}"), [("stocks.set_price_alert", {"symbol": qs, "price": thr, "direction": "above"})]),
        (c.say(f"đặt cảnh báo nếu {hs} rớt xuống dưới {c.money((hp // 1000 - 3) * 1000)}"),
         [("stocks.set_price_alert", {"symbol": hs, "price": (hp // 1000 - 3) * 1000, "direction": "below"})]),
        (c.say(f"nếu {qs} đang dưới {c.money(cond_lo)} thì mua {k} cổ, không thì thôi"),
         [("stocks.get_quote", {"symbol": qs}), ("stocks.place_order", {"symbol": qs, "side": "buy", "quantity": k})]),
        (c.say(f"nếu {qs} đang dưới {c.money((qp // 1000 - 5) * 1000)} thì mua {k} cổ, không thì thôi"),
         [("stocks.get_quote", {"symbol": qs})]),
        (c.say(f"mua thêm cổ phiếu {qs} cho tôi"), [("sys.ask_missing_info", {"tool_name": "stocks.place_order", "params": ["quantity"]})]),
    ]
    return st, tasks


def gen_translator(c):
    phrases = ["cảm ơn bạn rất nhiều", "nhà vệ sinh ở đâu", "món này có cay không", "cho tôi xin hoá đơn", "tôi bị dị ứng hải sản",
               "chúc mừng sinh nhật", "rất vui được gặp bạn", "bao nhiêu tiền một ký", "tôi muốn đặt bàn cho bốn người",
               "làm ơn gọi taxi giúp tôi", "ga tàu gần nhất ở đâu", "hẹn gặp lại vào tuần sau"]
    langs = [("en", "tiếng Anh"), ("ja", "tiếng Nhật"), ("ko", "tiếng Hàn"), ("zh", "tiếng Trung"), ("fr", "tiếng Pháp")]
    p, (code, lname) = c.pick(phrases), c.pick(langs)
    st = {"dictionary": [{"text": p, "to_lang": code, "translation": f"[{code}] {p}"}] if c.r.random() < 0.5 else []}
    tasks = [
        (c.say(f"dịch câu \"{p}\" sang {lname}"), [("translator.translate", {"text": p, "to_lang": code})]),
        (c.say(f"{p} {c.pick(['nói', 'tiếng'])} {lname.replace('tiếng ', '')} là gì"), [("translator.translate", {"text": p, "to_lang": code})]),
        (c.say(f"câu \"{p}\" {lname} nói thế nào"), [("translator.translate", {"text": p, "to_lang": code})]),
        (c.say(f"dịch giúp tôi câu \"{p}\""), [("sys.ask_missing_info", {"tool_name": "translator.translate", "params": ["to_lang"]})]),
    ]
    return st, tasks


def _products(c, pool, prefix, extra):
    out = []
    for i, (name, cat, price) in enumerate(pool):
        p = {"id": f"{prefix}{c.r.randint(100, 999)}{i}", "name": name, "price": price, "category": cat, **extra(name)}
        out.append(p)
    return out


def gen_grocery(c):
    pool = [("Sữa tươi Vinamilk 1L", "sữa", 32000), ("Trứng gà ta vỉ 10 quả", "trứng", 38000), ("Rau muống 500g", "rau", 12000),
            ("Thịt ba chỉ heo 500g", "thịt", 79000), ("Gạo ST25 túi 5kg", "gạo", 185000), ("Cà chua 1kg", "rau củ", 29000),
            ("Nước mắm Nam Ngư 500ml", "gia vị", 41000), ("Chuối già nải", "trái cây", 25000), ("Đậu phụ 2 bìa", "đậu", 10000),
            ("Cá basa phi lê 500g", "cá", 65000), ("Mì gói Hảo Hảo thùng 30 gói", "mì", 118000), ("Táo Envy 1kg", "trái cây", 159000)]
    c.r.shuffle(pool)
    prods = _products(c, pool[:7], "g", lambda n: {"unit": "gói" if "gói" in n else "phần", "in_stock": True})
    st = {"products": prods, "cart": [], "orders": []}
    a, b = prods[0], prods[1]
    qa, qb = c.r.randint(1, 4), c.r.randint(1, 3)
    ka = a["name"].split()[0].lower() + " " + a["name"].split()[1].lower()
    kb = b["name"].split()[0].lower() + " " + b["name"].split()[1].lower()
    h = max(c.r.randint(17, 19), min(21, c.now.hour + 2))
    tasks = [
        (c.say(f"tìm {ka} trên app đi chợ xem bao nhiêu tiền"), [("grocery.search", {"query": ka})]),
        (c.say(f"cho {c.n(qa)} {ka} vào giỏ"), [("grocery.search", {"query": ka}), ("grocery.add_to_cart", {"product_id": a["id"], "quantity": qa})]),
        (c.say(f"mua {c.n(qa)} {ka} với {c.n(qb)} {kb} rồi đặt hàng luôn"),
         [("grocery.search", {"query": ka}), ("grocery.add_to_cart", {"product_id": a["id"], "quantity": qa}),
          ("grocery.search", {"query": kb}), ("grocery.add_to_cart", {"product_id": b["id"], "quantity": qb}), ("grocery.checkout", {})]),
        (c.say(f"đặt {c.n(qa)} {ka}, giao lúc {hour_words(c, h)} nay"),
         [("grocery.search", {"query": ka}), ("grocery.add_to_cart", {"product_id": a["id"], "quantity": qa}),
          ("grocery.checkout", {"delivery_time": c.at(0, h)})]),
        (c.say(f"mua {ka} trên app đi chợ"), [("sys.ask_missing_info", {"tool_name": "grocery.add_to_cart", "params": ["quantity"]})]),
    ]
    return st, tasks


def gen_ecommerce(c):
    pool = [("Tai nghe Bluetooth Sony WH-CH520", "tai nghe", 990000), ("Tai nghe Xiaomi Redmi Buds 4", "tai nghe", 450000),
            ("Sạc dự phòng Anker 10000mAh", "sạc", 520000), ("Bình giữ nhiệt Lock&Lock 500ml", "gia dụng", 289000),
            ("Chuột không dây Logitech M331", "chuột", 349000), ("Áo khoác gió nam", "thời trang", 259000),
            ("Nồi chiên không dầu Philips 4.1L", "gia dụng", 2190000), ("Bàn phím cơ Akko 3068", "bàn phím", 1290000)]
    c.r.shuffle(pool)
    prods = _products(c, pool[:6], "sp", lambda n: {"seller": c.pick(["Shop Chính Hãng", "Mall Official", "Tiệm Nhà Mèo"]),
                                                      "rating": round(c.r.uniform(4.0, 5.0), 1), "in_stock": True})
    d_old = (c.now - timedelta(days=c.r.randint(3, 9))).isoformat()
    orders = [{"id": f"dh_{c.r.randint(1000, 9999)}", "items": [{"name": pool[6][0], "quantity": 1, "price": pool[6][2]}], "total": pool[6][2],
               "status": "shipping", "placed_at": d_old},
              {"id": f"dh_{c.r.randint(1000, 9999)}", "items": [{"name": pool[7][0], "quantity": 1, "price": pool[7][2]}], "total": pool[7][2],
               "status": "delivered", "placed_at": d_old, "delivered_at": (c.now - timedelta(days=1)).isoformat()}]
    st = {"products": prods, "cart": [], "orders": orders, "returns": []}
    p = prods[0]
    kw = p["category"]
    cap = (p["price"] // 100000 + 1) * 100000
    reason, rword = c.pick([("damaged", "bị vỡ"), ("damaged", "bị hỏng"), ("wrong_item", "giao sai mẫu"), ("changed_mind", "tôi không cần nữa")])
    sh, dl = orders[0], orders[1]
    sh_kw = sh["items"][0]["name"].split()[0].lower() + " " + sh["items"][0]["name"].split()[1].lower()
    dl_kw = dl["items"][0]["name"].split()[0].lower() + " " + dl["items"][0]["name"].split()[1].lower()
    pay, pword = c.pick([("ewallet", "ví điện tử"), ("bank", "chuyển khoản ngân hàng"), ("cod", "tiền mặt khi nhận hàng")])
    tasks = [
        (c.say(f"tìm {kw} dưới {c.money(cap)} trên sàn"), [("ecommerce.search", {"query": kw, "max_price": cap})]),
        (c.say(f"đơn hàng của tôi giao tới đâu rồi"), [("ecommerce.order_status", {})]),
        (c.say(f"huỷ đơn {sh_kw} tôi đặt hôm trước"), [("ecommerce.order_status", {}), ("ecommerce.cancel_order", {"order_id": sh["id"]})]),
        (c.say(f"cái {dl_kw} mới giao {rword}, làm yêu cầu trả hàng"),
         [("ecommerce.order_status", {}), ("ecommerce.request_return", {"order_id": dl["id"], "reason": reason})]),
        (c.say(f"thêm {c.n(1)} {kw} {p['name'].split()[-1]} vào giỏ rồi thanh toán bằng {pword}"),
         [("ecommerce.search", {"query": f"{kw} {p['name'].split()[-1]}"}), ("ecommerce.add_to_cart", {"product_id": p["id"], "quantity": 1}),
          ("ecommerce.checkout", {"payment": pay})]),
        (c.say(f"trả lại cái {dl_kw} mới nhận"), [("ecommerce.order_status", {}),
                                                ("sys.ask_missing_info", {"tool_name": "ecommerce.request_return", "params": ["reason"]})]),
    ]
    return st, tasks


def gen_event_tickets(c):
    pool = [("Đêm nhạc Hà Anh Tuấn - Chân trời rực rỡ", "ca nhạc", "Nhà hát Hoà Bình", 1200000), ("Kịch Bí mật vườn Lệ Chi", "kịch", "Nhà hát Lớn", 350000),
            ("Liveshow Mỹ Tâm", "ca nhạc", "Sân vận động Quân khu 7", 1500000), ("Triển lãm tranh Đông Hồ", "triển lãm", "Bảo tàng Mỹ thuật", 80000),
            ("Chung kết bóng chuyền nữ", "thể thao", "Nhà thi đấu Phú Thọ", 200000), ("Hài kịch Tết Thanh Xuân", "kịch", "Sân khấu Idecaf", 450000),
            ("Giải chạy đêm Hồ Tây", "thể thao", "Hồ Tây", 300000)]
    c.r.shuffle(pool)
    evs = []
    for i, (n, cat, v, p) in enumerate(pool[:5]):
        off = c.r.randint(2, 20)
        evs.append({"id": f"ev_{c.r.randint(10, 99)}{i}", "name": n, "category": cat, "date": (c.now + timedelta(days=off)).date().isoformat(),
                    "time": c.pick(["19:30", "20:00", "18:00", "09:00"]), "venue": v, "price": p, "available": c.r.randint(5, 60)})
    st = {"events": evs, "bookings": []}
    e = evs[0]
    key = c.pick([e["name"].split(" - ")[0].replace("Đêm nhạc ", "").replace("Liveshow ", ""), e["category"]])
    hits = [x for x in evs if key.lower() in (x["name"] + x["category"] + x["venue"]).lower()]
    q = c.r.randint(1, 4)
    tasks = [
        (c.say(f"{c.pick(['dạo này có', 'tìm giúp'])} sự kiện {e['category']} nào không"), [("event_tickets.search", {"query": e["category"]})]),
        (c.say("có sự kiện gì sắp diễn ra không"), [("event_tickets.search", {})]),
    ]
    if len(hits) == 1:
        tasks.append((c.say(f"đặt {c.n(q)} vé {key}"), [("event_tickets.search", {"query": key}), ("event_tickets.book", {"event_id": e["id"], "quantity": q})]))
    else:
        tasks.append((c.say(f"đặt {c.n(q)} vé {key}"), [("event_tickets.search", {"query": key}),
                                                       ("sys.ask_missing_info", {"tool_name": "event_tickets.book", "params": ["event_id"]})]))
    tasks.append((c.say(f"mua vé {e['name'].split(' - ')[0]}"), [("sys.ask_missing_info", {"tool_name": "event_tickets.book", "params": ["quantity"]})]))
    return st, tasks


def gen_habit_tracker(c):
    pool = [("Chạy bộ", "chạy bộ"), ("Thiền 10 phút", "thiền"), ("Đọc sách", "đọc sách"), ("Uống đủ 2 lít nước", "uống nước"),
            ("Học tiếng Anh", "học tiếng Anh"), ("Tập yoga", "tập yoga"), ("Đi ngủ trước 23h", "ngủ sớm")]
    c.r.shuffle(pool)
    y = (c.now - timedelta(days=1)).date().isoformat()
    habits = [{"id": f"hb_{i + 1:02d}", "name": n, "streak": c.r.randint(0, 25), "last_logged": c.pick([y, (c.now - timedelta(days=3)).date().isoformat()])}
              for i, (n, _) in enumerate(pool[:4])]
    st = {"habits": habits, "logs": []}
    h, word = habits[0], pool[0][1]
    today, yday = c.now.date().isoformat(), y
    tasks = [
        (c.say(f"hôm nay tôi {word} rồi, đánh dấu giúp"), [("habit_tracker.list_habits", {}), ("habit_tracker.log", {"habit_id": h["id"], "date": today})]),
        (c.say(f"ghi là hôm qua tôi có {word} nhé"), [("habit_tracker.list_habits", {}), ("habit_tracker.log", {"habit_id": h["id"], "date": yday})]),
        (c.say(f"chuỗi {word} của tôi được mấy ngày rồi"), [("habit_tracker.list_habits", {})]),
        (c.say(f"hôm nay xong {word} với {pool[1][1]} rồi"),
         [("habit_tracker.list_habits", {}), ("habit_tracker.log", {"habit_id": h["id"], "date": today}),
          ("habit_tracker.log", {"habit_id": habits[1]["id"], "date": today})]),
    ]
    return st, tasks


def gen_pantry(c):
    pool = [("Trứng gà", "quả"), ("Gạo", "kg"), ("Mì gói", "gói"), ("Sữa hộp", "hộp"), ("Hành tây", "củ"), ("Khoai tây", "củ"),
            ("Dầu ăn", "chai"), ("Cà chua", "quả"), ("Bột mì", "kg"), ("Xúc xích", "cây")]
    c.r.shuffle(pool)
    items = [{"id": f"pt_{i + 1:02d}", "name": n, "stock": c.r.randint(3, 20), "unit": u} for i, (n, u) in enumerate(pool[:6])]
    st = {"items": items}
    a, b = items[0], items[1]
    qa = c.r.randint(1, min(3, a["stock"]))
    qb = c.r.randint(1, min(2, b["stock"]))
    tasks = [
        (c.say(f"trong bếp còn bao nhiêu {a['name'].lower()}"), [("pantry.list_items", {})]),
        (c.say(f"tôi vừa dùng {c.n(qa)} {a['unit']} {a['name'].lower()}, trừ đi"), [("pantry.list_items", {}), ("pantry.consume", {"item_id": a["id"], "quantity": qa})]),
        (c.say(f"nấu xong rồi, ghi là đã dùng {c.n(qa)} {a['unit']} {a['name'].lower()} với {c.n(qb)} {b['unit']} {b['name'].lower()}"),
         [("pantry.list_items", {}), ("pantry.consume", {"item_id": a["id"], "quantity": qa}), ("pantry.consume", {"item_id": b["id"], "quantity": qb})]),
        (c.say(f"đánh dấu là tôi đã dùng {a['name'].lower()}"), [("pantry.list_items", {}),
                                                               ("sys.ask_missing_info", {"tool_name": "pantry.consume", "params": ["quantity"]})]),
    ]
    return st, tasks


def gen_parking(c):
    pool = [("Bãi xe Bờ Hồ", "Đinh Tiên Hoàng, Hoàn Kiếm", "Hồ Gươm"), ("Bãi đỗ Vincom Bà Triệu", "191 Bà Triệu", "Hai Bà Trưng"),
            ("Bãi xe sân bay Tân Sơn Nhất", "Trường Sơn, Tân Bình", "Tân Sơn Nhất"), ("Bãi đỗ chợ Bến Thành", "Lê Lợi, Quận 1", "Bến Thành"),
            ("Bãi xe Cầu Rồng", "Bạch Đằng, Hải Châu", "Cầu Rồng"), ("Bãi đỗ Aeon Mall Long Biên", "27 Cổ Linh", "Long Biên")]
    c.r.shuffle(pool)
    lots = [{"id": f"lot_{c.r.randint(10, 99)}{i}", "name": n, "address": a, "area": ar, "price_per_hour": c.pick([20000, 25000, 30000, 15000]),
             "available_spots": c.r.randint(1, 40)} for i, (n, a, ar) in enumerate(pool[:4])]
    h0 = min(18, max(6, c.now.hour - 2))
    sess = [{"id": "park_01", "lot_id": lots[1]["id"], "from_time": c.at(0, h0), "hours": 3,
             "end_time": c.at(0, h0 + 3), "total": lots[1]["price_per_hour"] * 3, "status": "active"}]
    st = {"lots": lots, "sessions": sess}
    lot = lots[0]
    h, hrs = c.r.randint(8, 19), c.r.randint(1, 4)
    off, dword = c.pick([(0, "hôm nay"), (1, "mai")])
    if off == 0 and h <= c.now.hour:
        off, dword = 1, "mai"
    tasks = [
        (c.say(f"tìm bãi đỗ xe gần {lot['area']}"), [("parking.find", {"near": lot["area"]})]),
        (c.say(f"đặt chỗ đỗ xe ở {lot['area']} lúc {hour_words(c, h)} {dword}, {c.n(hrs)} tiếng"),
         [("parking.find", {"near": lot["area"]}), ("parking.reserve", {"lot_id": lot["id"], "from_time": c.at(off, h), "hours": hrs})]),
        (c.say("tôi lấy xe rồi, kết thúc phiên đỗ đi"), [("parking.list_sessions", {}), ("parking.end_session", {"session_id": "park_01"})]),
        (c.say("xem tôi đang có chỗ đỗ nào đã đặt"), [("parking.list_sessions", {})]),
        (c.say(f"đặt chỗ đỗ gần {lot['area']} {dword}"), [("sys.ask_missing_info", {"tool_name": "parking.reserve", "params": ["from_time", "hours"]})]),
    ]
    return st, tasks


def gen_public_transit(c):
    lines = [("32", ["Giáp Bát", "Kim Liên", "Cầu Giấy", "Nhổn"]), ("Metro 2A", ["Cát Linh", "Láng", "Thái Hà", "Hà Đông"]),
             ("09", ["Bờ Hồ", "Tràng Thi", "Đại học Bách Khoa", "Giáp Bát"]), ("Metro số 1", ["Bến Thành", "Ba Son", "Thảo Điền", "Suối Tiên"]),
             ("19", ["Bến Thành", "Chợ Lớn", "Đầm Sen"])]
    c.r.shuffle(lines)
    routes = [{"id": f"r_{i + 1:02d}", "name": f"Tuyến {n}" if not n.startswith("Metro") else n, "type": "metro" if n.startswith("Metro") else "bus",
               "stops": s, "fare": 7000 if not n.startswith("Metro") else 15000, "duration_min": c.r.randint(25, 60), "headway_min": c.pick([10, 15, 20])}
              for i, (n, s) in enumerate(lines[:3])]
    deps = []
    for r in routes:
        for k in range(3):
            deps.append({"stop": r["stops"][1], "route": r["id"], "time": c.at(0, min(22, c.now.hour + 1), (k * 15) % 60)})
    st = {"card_balance": c.pick([12000, 35000, 50000, 80000]), "routes": routes, "departures": deps, "topups": []}
    r = routes[0]
    o, d = r["stops"][0], r["stops"][-1]
    amt = c.pick([50000, 100000, 200000])
    num = r["name"].replace("Tuyến ", "")
    tasks = [
        (c.say(f"đi xe buýt từ {o} tới {d} thì đi tuyến nào"), [("public_transit.find_routes", {"origin": o, "destination": d})]),
        (c.say(f"từ {o} đến {d} đi {c.pick(['metro', 'xe buýt'])} được không"), [("public_transit.find_routes", {"origin": o, "destination": d})]),
        (c.say(f"chuyến tiếp theo ở trạm {r['stops'][1]} mấy giờ"), [("public_transit.next_departures", {"stop": r["stops"][1]})]),
        (c.say(f"xe {num} ở trạm {r['stops'][1]} mấy giờ có chuyến"), [("public_transit.next_departures", {"stop": r["stops"][1], "route": num})]),
        (c.say(f"nạp {c.money(amt)} vào thẻ xe buýt"), [("public_transit.topup_card", {"amount": amt})]),
        (c.say("nạp tiền vào thẻ đi xe giúp tôi"), [("sys.ask_missing_info", {"tool_name": "public_transit.topup_card", "params": ["amount"]})]),
    ]
    return st, tasks


def gen_recipes(c):
    pool = [("Canh chua cá lóc", ["canh chua", "cá"], [("Cá lóc", 500, "g"), ("Me chua", 50, "g"), ("Cà chua", 2, "quả"), ("Dứa", 0.25, "quả")]),
            ("Súp bí đỏ", ["súp", "bí đỏ"], [("Bí đỏ", 400, "g"), ("Sữa tươi", 200, "ml"), ("Hành tây", 1, "củ")]),
            ("Thịt kho trứng", ["thịt kho", "trứng"], [("Thịt ba chỉ", 500, "g"), ("Trứng vịt", 6, "quả"), ("Nước dừa", 400, "ml")]),
            ("Phở bò", ["phở", "bò"], [("Bánh phở", 500, "g"), ("Thịt bò", 300, "g"), ("Xương bò", 1, "kg")]),
            ("Gà xào sả ớt", ["gà", "sả ớt"], [("Thịt gà", 500, "g"), ("Sả", 4, "cây"), ("Ớt", 2, "quả")]),
            ("Bánh xèo", ["bánh xèo", "tôm"], [("Bột gạo", 300, "g"), ("Tôm", 200, "g"), ("Giá đỗ", 200, "g")])]
    c.r.shuffle(pool)
    recs = [{"id": f"rc_{c.r.randint(10, 99)}{i}", "name": n, "tags": t, "servings": 2,
             "ingredients": [{"name": a, "quantity": q, "unit": u} for a, q, u in ing], "steps": ["Sơ chế nguyên liệu", "Nấu chín", "Nêm nếm vừa ăn"]}
            for i, (n, t, ing) in enumerate(pool[:4])]
    st = {"recipes": recs, "saved": []}
    r = recs[0]
    name = r["name"].lower()
    ppl = c.r.randint(3, 6)
    tasks = [
        (c.say(f"tìm cách nấu {name}"), [("recipes.search", {"query": name})]),
        (c.say(f"nấu {name} cho {c.n(ppl)} người thì cần những gì"), [("recipes.search", {"query": name, "servings": ppl})]),
        (c.say(f"lưu công thức {name} lại"), [("recipes.search", {"query": name}), ("recipes.save", {"recipe_id": r["id"]})]),
        (c.say(f"chỉ tôi các bước làm {name}"), [("recipes.search", {"query": name}), ("recipes.get", {"recipe_id": r["id"]})]),
    ]
    return st, tasks


def gen_train_bus_tickets(c):
    city = c.pick([("Hà Nội", "Hải Phòng"), ("Hà Nội", "Vinh"), ("TP.HCM", "Đà Lạt"), ("TP.HCM", "Nha Trang"), ("Đà Nẵng", "Huế"), ("Hà Nội", "Lào Cai")])
    o, d = city if c.r.random() < 0.7 else city[::-1]
    off = c.r.randint(1, 9)
    date, dword = c.day(off)
    trips = []
    for i in range(4):
        typ = c.pick(["train", "bus"])
        h = c.pick([6, 8, 13, 21])
        trips.append({"id": f"tr_{c.r.randint(100, 999)}{i}", "type": typ, "operator": "Đường sắt VN" if typ == "train" else c.pick(["Phương Trang", "Hoàng Long", "Hải Âu"]),
                      "origin": o, "destination": d, "date": date if i < 3 else c.day(off + 1)[0], "depart": f"{h:02d}:00", "arrive": f"{(h + 5) % 24:02d}:30",
                      "price": c.pick([180000, 250000, 320000, 450000]), "seats_left": c.r.randint(2, 30)})
    old = trips[3]
    st = {"trips": trips, "bookings": [{"id": "tk_01", "trip_id": old["id"], "seats": 2, "total": old["price"] * 2, "status": "confirmed"}]}
    t = trips[0]
    s = c.r.randint(1, 3)
    kind = "tàu" if t["type"] == "train" else "xe khách"
    hits = [x for x in trips if x["date"] == date and x["type"] == t["type"]]
    tasks = [
        (c.say(f"tìm vé đi từ {o} ra {d} {dword}"), [("train_bus_tickets.search", {"origin": o, "destination": d, "date": date})]),
        (c.say(f"{dword} có chuyến {kind} nào từ {o} đi {d} không"),
         [("train_bus_tickets.search", {"origin": o, "destination": d, "date": date, "type": t["type"]})]),
        (c.say("xem các vé tàu xe tôi đã đặt"), [("train_bus_tickets.my_bookings", {})]),
        (c.say("huỷ vé xe tôi đã đặt"), [("train_bus_tickets.my_bookings", {}), ("train_bus_tickets.cancel", {"booking_id": "tk_01"})]),
        (c.say(f"tìm vé {kind} đi {d}"), [("sys.ask_missing_info", {"tool_name": "train_bus_tickets.search", "params": ["date"]})]),
    ]
    if len(hits) == 1:
        tasks.append((c.say(f"đặt {c.n(s)} vé {kind} từ {o} đi {d} {dword}"),
                      [("train_bus_tickets.search", {"origin": o, "destination": d, "date": date, "type": t["type"]}),
                       ("train_bus_tickets.book", {"trip_id": t["id"], "seats": s})]))
    return st, tasks


GENS = {"stocks": gen_stocks, "translator": gen_translator, "grocery": gen_grocery, "ecommerce": gen_ecommerce,
        "event_tickets": gen_event_tickets, "habit_tracker": gen_habit_tracker, "pantry": gen_pantry, "parking": gen_parking,
        "public_transit": gen_public_transit, "recipes": gen_recipes, "train_bus_tickets": gen_train_bus_tickets}


def combo(c, item, states, utter, calls):
    """Sometimes chain a seen-app action the base sample supports: a reminder or an SMS to a contact."""
    tools = {t for a in item["context"]["installed_apps"].values() for t in a["tools"]}
    if c.r.random() < 0.5 and "reminders.create" in tools:
        h = c.r.randint(7, 20)
        what = c.pick(["đi lấy hàng", "kiểm tra lại đơn", "chuẩn bị đồ", "ra bến xe", "nấu cơm"])
        return utter.rstrip(".!") + f", rồi nhắc tôi {what} lúc {hour_words(c, h)} mai.", calls + [
            ("reminders.create", {"content": what[0].upper() + what[1:], "time": c.at(1, h)})]
    cons = states.get("contacts", {}).get("contacts", [])
    if cons and "contacts.search" in tools and "sms.send" in tools:
        p = c.pick(cons)
        msg = c.pick(["xong rồi nhé", "mình đặt rồi đó", "tối nay về muộn chút", "nhớ mang theo áo mưa"])
        return utter.rstrip(".!") + f", xong nhắn {p['name']} là {msg}.", calls + [
            ("contacts.search", {"query": p["name"]}), ("sms.send", {"contact_id": p["id"], "content": msg})]
    return utter, calls


def build(i, rng, base_items, envs, catalog, app):
    base = rng.choice(base_items)
    item = {"id": f"synth_{i:04d}", "audio": None, "history": [], "context": copy.deepcopy(base["context"])}
    states = {a: copy.deepcopy(s) for a, s in envs[base["id"]].items() if a != "_meta"}
    now = datetime.fromisoformat(item["context"]["current_time"])
    c = Ctx(rng, now)
    st, tasks = GENS[app](c)
    utter, plan = rng.choice(tasks)
    tools = [t["name"] for a in catalog["apps"] if a["id"] == app for t in a["tools"]]
    installed = rng.random() > 0.08  # a few requests for an app that is not installed: gold is no call at all
    if installed:
        item["context"]["installed_apps"][app] = {"tools": tools}
        states[app] = st
        if plan and plan[-1][0] != "sys.ask_missing_info" and rng.random() < 0.3:
            utter, plan = combo(c, item, states, utter, plan)
    else:
        plan = []
    sim = Simulator({"context": {"current_time": item["context"]["current_time"],
                                 "installed_apps": {a: {"tools": v["tools"], "state": states.get(a, {})} for a, v in item["context"]["installed_apps"].items()}}},
                    catalog)
    results = []
    for name, args in plan:
        seen = json.dumps(results, ensure_ascii=False)
        if any(k.endswith("_id") and isinstance(v, str) and f'"{v}"' not in seen for k, v in args.items()):
            return None  # an ID the model could not have read from an earlier result
        r = sim.call(name, args)
        if name == "contacts.search" and len(r.get("contacts", [])) != 1:
            return None  # several (or no) people match: gold would ask instead
        if not r.get("ok") or r.get("done") is False:
            return None  # template produced a call the engine refuses or ignores: drop the sample
        results.append(r)
    item["audio_text"] = utter
    item["label"] = {"tool_calls": [{"name": n, "arguments": a} for n, a in plan], "tool_results": results}
    item["synth_app"] = app
    return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=int, default=200, help="train samples never used as a base (the eval holdout)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    items = [json.loads(l) for l in open(ROOT / "train/train.jsonl", encoding="utf-8")]
    held = {d["id"] for d in random.Random(0).sample(items, a.holdout)} if a.holdout else set()
    base = [d for d in items if d["id"] not in held]
    envs = {d["id"]: json.load(open(ROOT / "train/train_environment" / d["id"] / "state.json", encoding="utf-8")) for d in base}
    pub = json.load(open(ROOT / "public_test/public_test_tools_catalog.json", encoding="utf-8"))
    trn = json.load(open(ROOT / "train/train_tools_catalog.json", encoding="utf-8"))
    seen = {x["id"] for x in pub["apps"]}
    catalog = {**pub, "apps": pub["apps"] + [x for x in trn["apps"] if x["id"] not in seen]}

    rng = random.Random(a.seed)
    apps = sorted(GENS)
    out, dropped = [], 0
    while len(out) < a.n:
        it = build(len(out) + 1, rng, base, envs, catalog, apps[len(out) % len(apps)])
        if it is None:
            dropped += 1
            continue
        out.append(it)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    kinds = {"exec": 0, "ask": 0, "empty": 0}
    for it in out:
        calls = it["label"]["tool_calls"]
        kinds["empty" if not calls else "ask" if calls[-1]["name"] == "sys.ask_missing_info" else "exec"] += 1
    print(f"{len(out)} samples ({dropped} dropped by the engine check) -> {a.out}; {kinds}")


if __name__ == "__main__":
    main()
