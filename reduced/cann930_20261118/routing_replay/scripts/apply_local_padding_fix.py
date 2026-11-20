"""Apply the verified local-padding branch while preserving source line endings."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


source = Path("/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_routed_experts_capture.py")
patches = Path("/mnt/share/c00954197/Workspace/dsv41/basic/routing_replay/patches")
raw = source.read_bytes()
ending = "\r\n" if b"\r\n" in raw else "\n"
text = raw.decode().replace("\r\n", "\n")
if "and n == _padded_batch_rows(self.tp_size)" in text:
    print("Local padding patch already applied")
    raise SystemExit(0)
shutil.copy2(source, patches / "routed_capture.before.py")
anchor = "            end_loc = token_num_per_dp\n        else:\n            sp_expected ="
assert text.count(anchor) == 1
replacement = """            end_loc = token_num_per_dp
        elif (
            self.tp_size == 1
            and _EXTRA_CTX.moe_comm_type in _TP_SHARDED_COMM_TYPES
            and n > token_num_per_dp
            and n == _padded_batch_rows(self.tp_size)
        ):
            # Dispatch can pad a TP1 rank's local input, including DP dummy
            # forwards. Its declared target identifies this local layout.
            # Only this rank's unpadded rows belong in the capture buffer.
            start_loc = 0
            end_loc = token_num_per_dp
        else:
            sp_expected ="""
text = text.replace(anchor, replacement)
source.write_bytes(text.replace("\n", ending).encode())
shutil.copy2(source, patches / "routed_capture.after.py")
diff = subprocess.check_output(
    ["git", "-C", "/vllm-workspace/vllm-ascend", "diff", "--no-ext-diff", "--",
     "vllm_ascend/patch/worker/patch_routed_experts_capture.py"], text=True
)
(patches / "local_dp_padding_capture.patch").write_text(diff)
print(json.dumps({"before_sha256": hashlib.sha256(raw).hexdigest(),
                  "after_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "diff_lines": len(diff.splitlines())}))
