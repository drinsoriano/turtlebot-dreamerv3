import sys
import torch


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f": {detail}" if detail else ""))
    return condition


ok = True
ok &= check("torch imports", True, f"version={torch.__version__}")
ok &= check("CUDA available", torch.cuda.is_available())
ok &= check(
    "GPU detected",
    torch.cuda.device_count() > 0,
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
)
try:
    t = torch.tensor([1.0]).cuda() + torch.tensor([2.0]).cuda()
    ok &= check("CUDA tensor op", t.item() == 3.0, f"result={t.item()}")
except Exception as e:
    ok &= check("CUDA tensor op", False, str(e))

print()
print("All checks PASSED" if ok else "One or more checks FAILED")
sys.exit(0 if ok else 1)
