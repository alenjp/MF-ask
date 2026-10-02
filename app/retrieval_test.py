import sys
from pathlib import Path

APP = str(Path(__file__).resolve().parent)
if APP not in sys.path:
    sys.path.insert(0, APP)

from rag import retrieve, detect_scheme, ask  # noqa: E402
from config import TOP_K, TOP_K_GENERIC  # noqa: E402

q = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else (TOP_K if detect_scheme(q) else TOP_K_GENERIC)
k = TOP_K if detect_scheme(q) else TOP_K_GENERIC

print(f"Q: {q}")
print(f"detect_scheme -> {detect_scheme(q)}")
print(f"adaptive k -> {k}")
hits = retrieve(q, k=k)
print(f"{'score':>7}  scheme / section")
print("-" * 72)
for h in hits:
    print(f"{h.score:7.3f}  {h.scheme} / {h.section}")
print()
a = ask(q)
print(f"[ask] kind={a.kind} | {a.text}")