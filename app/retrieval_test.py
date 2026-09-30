import sys

APP = r"C:\Users\User\build hour27 sept\app"
if APP not in sys.path:
    sys.path.insert(0, APP)

from rag import retrieve, detect_scheme, ask  # noqa: E402

q = sys.argv[1]
k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
k = k if detect_scheme(q) else 8

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