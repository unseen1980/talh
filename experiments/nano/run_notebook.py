"""
run_notebook.py — execute TALH_Colab.ipynb cells without a Jupyter kernel.

Reads each code cell, runs it with exec() in a shared namespace, and logs
output to run_notebook.log.  Stops on the first unhandled exception.

Training cells (§4.1, §4.2) are skipped when a checkpoint already exists at
checkpoints/talh_full/best.pt — avoids overwriting a Colab-trained checkpoint
with a slower/shorter local training run.
"""

import json
import sys
import time
import traceback
import io
import contextlib
from pathlib import Path

LOG = Path("run_notebook.log")
NB  = Path("TALH_Colab.ipynb")

# Cell indices (0-based over *code* cells only — markdown cells excluded).
# The 19 code cells map as: 0=§1.1, 1=§1.2, 2=§1.3, 3=§2.1, 4=§3.1,
#   5=§4.1(PyTorch train), 6=§4.2(all ablations), 7=§4.3(MLX train),
#   8=§5.1, 9=§5.2, 10=§6.1, ...
_TRAINING_CELL_INDICES = {5, 6, 7}  # §4.1, §4.2, §4.3
_CHECKPOINT_GUARD = Path("checkpoints/talh_full/best.pt")


def log(msg: str, also_print: bool = True) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    with LOG.open("a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


class Tee(io.TextIOBase):
    """Write to both a real stream and the log file."""
    def __init__(self, stream):
        self._s = stream

    def write(self, s):
        with LOG.open("a") as f:
            f.write(s)
        return self._s.write(s)

    def flush(self):
        self._s.flush()


def run_all() -> None:
    LOG.write_text("")   # clear log
    log(f"=== run_notebook.py starting: {NB} ===")

    nb = json.loads(NB.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    log(f"Found {len(cells)} code cells")

    # Shared execution namespace — mirrors what Jupyter does
    ns: dict = {
        "__name__": "__main__",
        "__file__":  str(NB.resolve()),
    }

    # Redirect stdout/stderr through Tee so output goes to log AND terminal
    sys.stdout = Tee(sys.__stdout__)
    sys.stderr = Tee(sys.__stderr__)

    skip_training = _CHECKPOINT_GUARD.exists()
    if skip_training:
        log(f"Checkpoint found at {_CHECKPOINT_GUARD} — skipping training cells {_TRAINING_CELL_INDICES}")

    for idx, cell in enumerate(cells):
        cell_id  = cell.get("id", f"cell_{idx}")
        src      = "".join(cell["source"]).strip()
        if not src:
            continue

        if skip_training and idx in _TRAINING_CELL_INDICES:
            log(f"\n--- Cell {idx} [{cell_id}]: [SKIPPED — checkpoint exists] ---")
            continue

        # Print first line as a preview
        preview = src.splitlines()[0][:80]
        log(f"\n--- Cell {idx} [{cell_id}]: {preview} ---", also_print=True)

        try:
            exec(compile(src, f"<cell {idx}>", "exec"), ns)  # noqa: S102
        except SystemExit as e:
            log(f"Cell {idx} called sys.exit({e.code}) — stopping.")
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            sys.exit(e.code)
        except Exception:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log(f"\n!!! ERROR in cell {idx} [{cell_id}] !!!")
            log(traceback.format_exc())
            log("Stopping. Fix the error and re-run.")
            sys.exit(1)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log("\n=== All cells completed successfully ===")


if __name__ == "__main__":
    run_all()
