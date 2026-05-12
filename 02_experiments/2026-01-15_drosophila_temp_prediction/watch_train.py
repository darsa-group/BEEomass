#!/usr/bin/env python3
"""Tail train.log and display training stats in a clean table."""
import re
import sys
import time

LOG = "train.log"

HDR = f"{'Epoch':>6}  {'Tr Loss':>8}  {'Tr Sex':>7}  {'Tr Sp':>7}  {'Tr T':>7}  {'Tr All':>7}  |  {'Val Loss':>8}  {'Val Sex':>7}  {'Val Sp':>7}  {'Val T':>7}  {'Val All':>7}"
SEP = "-" * len(HDR)

RE_DONE = re.compile(
    r"Epoch\s+(\d+)\s+done\s*\|"
    r"\s*train loss=([\d.]+)\s+sex=([\d.]+)\s+sp=([\d.]+)\s+t=([\d.]+)\s+all=([\d.]+)"
    r"\s*\|\s*val loss=([\d.]+)\s+sex=([\d.]+)\s+sp=([\d.]+)\s+t=([\d.]+)\s+all=([\d.]+)"
)
RE_STEP = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\[(\d+)/(\d+)\].*?loss=([\d.]+).*?acc\(sex=([\d.]+),\s*sp=([\d.]+),\s*t=([\d.]+),\s*all=([\d.]+)\)\s*([\d.]+)\s*it/s"
)

def main():
    printed_header = False
    epochs_shown = set()

    with open(LOG, "r") as f:
        f.seek(0)
        buf = ""
        while True:
            chunk = f.read(4096)
            if chunk:
                buf += chunk
                lines = re.split(r"[\r\n]", buf)
                buf = lines[-1]  # keep incomplete last segment

                last_step = None
                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue

                    m = RE_DONE.search(line)
                    if m:
                        ep = int(m.group(1))
                        if ep not in epochs_shown:
                            epochs_shown.add(ep)
                            if not printed_header:
                                print(HDR)
                                print(SEP)
                                printed_header = True
                            else:
                                # clear the step line
                                print("\r" + " " * 120 + "\r", end="")
                            tl, ts, tsp, tt, ta = m.group(2,3,4,5,6)
                            vl, vs, vsp, vt, va = m.group(7,8,9,10,11)
                            print(f"{ep:>6}  {float(tl):>8.4f}  {float(ts):>7.3f}  {float(tsp):>7.3f}  {float(tt):>7.3f}  {float(ta):>7.3f}  |  {float(vl):>8.4f}  {float(vs):>7.3f}  {float(vsp):>7.3f}  {float(vt):>7.3f}  {float(va):>7.3f}")
                        continue

                    m = RE_STEP.search(line)
                    if m:
                        last_step = m

                if last_step and printed_header:
                    ep, tot_ep, step, tot_step = (int(x) for x in last_step.group(1,2,3,4))
                    loss, s_acc, sp_acc, t_acc, a_acc, its = (
                        float(last_step.group(x)) for x in (5,6,7,8,9,10)
                    )
                    bar_width = 20
                    filled = int(bar_width * step / max(tot_step, 1))
                    bar = "#" * filled + "." * (bar_width - filled)
                    msg = (f"  Ep {ep:03d}/{tot_ep} [{bar}] {step}/{tot_step}  "
                           f"loss={loss:.4f}  sex={s_acc:.3f}  sp={sp_acc:.3f}  t={t_acc:.3f}  all={a_acc:.3f}  {its:.1f} it/s")
                    print(f"\r{msg:<120}", end="", flush=True)

            else:
                time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
