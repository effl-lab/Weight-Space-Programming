"""Builds the qualitative-reasoning weights from qr.txt (see the header of that file)."""
import re
from pathlib import Path

import torch

from textproc import porter

HERE = Path(__file__).resolve().parent
KIND = {"Q": 0, "E": 1, "I": 2}


def parse():
    names, trig, amt, fwd, rate, rels, zero = {}, [], {}, {}, {}, [], {}
    for line in (HERE / "qr.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^!\s*:\s*(.*)$", line)
        if m:
            for wd in m.group(1).split():
                zero["!" + wd] = 1.0
            continue
        m = re.match(r"^=\s*:\s*(.*)$", line)
        if m:
            for wd in m.group(1).split():
                zero[wd] = 1.0
            continue
        m = re.match(r"^(\+>|->|\+|-|r\+|r-)\s*:\s*(.*)$", line)
        if m:
            tag = m.group(1)
            sign = 1.0 if "+" in tag else -1.0
            table = rate if tag.startswith("r") else fwd if tag.endswith(">") else amt
            for wd in m.group(2).split():
                table[wd] = sign
            continue
        m = re.match(r"^([QEI])\s+(\w+)\s*([+-]?)\s*:\s*(.*)$", line)
        if m:
            kind, name, sg, ws = m.groups()
            names.setdefault(name, len(names) + 1)
            trig.append((name, KIND[kind], -1.0 if sg == "-" else 1.0, ws.split()))
            continue
        m = re.match(r"^R\s+(.*?)\s*([<>])\s*(.*)$", line)
        if m:
            rels.append((m.group(1).split(), 1.0 if m.group(2) == ">" else -1.0, m.group(3).split()))
            continue
        raise ValueError("qr.txt: cannot parse " + line)
    return names, trig, amt, fwd, rate, rels, zero


def build_qr(vocab):
    names, trig, amt, fwd, rate, rels, zero = parse()
    Qn = len(names) + 1
    by_exact, by_stem = {}, {}
    for name, kind, sg, ws in trig:
        for wd in ws:
            for table, key in ((by_exact, wd), (by_stem, porter(wd))):
                lst = table.setdefault(key, [])
                if all(r[0] != names[name] for r in lst):
                    lst.append((names[name], kind, sg))
    V = len(vocab) + 2
    qa = torch.zeros(V, dtype=torch.long)
    qb = torch.zeros(V, dtype=torch.long)
    sa, sb, ka, kb = (torch.zeros(V) for _ in range(4))
    pa, pf, pr, pz = torch.zeros(V), torch.zeros(V), torch.zeros(V), torch.zeros(V)
    for i, wd in enumerate(vocab):
        base = wd[:-2] if wd.endswith("'s") else wd.rstrip("'")
        roles = by_exact.get(base) or by_stem.get(porter(base)) or []
        if "!" + base in zero:  # words that name no quantity although their stem does ('dissolved oxygen')
            roles = []
        if roles:
            qa[i + 2], ka[i + 2], sa[i + 2] = roles[0]
        if len(roles) > 1:
            qb[i + 2], kb[i + 2], sb[i + 2] = roles[1]
        pa[i + 2] = amt.get(base, 0.0)
        pf[i + 2] = fwd.get(base, 0.0)
        pr[i + 2] = rate.get(base, 0.0)
        pz[i + 2] = zero.get(base, 0.0)
    I = torch.zeros(Qn, Qn)
    for causes, sign, effects in rels:
        for c in causes:
            for e in effects:
                if c not in names or e not in names:
                    raise ValueError(f"qr.txt: unknown quantity in relation {c} -> {e}")
                if c != e:
                    I[names[c], names[e]] = sign
    # two-step chains (a -> b -> c) count half where there is no direct link
    I2 = torch.sign(I @ I) * (I == 0).float()
    I2.fill_diagonal_(0.0)
    I = I + 0.5 * I2
    return {"qr_self": torch.eye(Qn) * 0.5, "qr_a": qa, "qr_b": qb, "qr_sa": sa, "qr_sb": sb, "qr_ka": ka, "qr_kb": kb,
            "qr_amt": pa, "qr_fwd": pf, "qr_rate": pr, "qr_zero": pz, "qr_I": I}
