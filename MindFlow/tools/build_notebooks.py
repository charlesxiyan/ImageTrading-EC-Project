"""Builds the MindFlow notebooks.  Usage: python MindFlow/tools/build_notebooks.py [output folder]

One notebook per step of the project's reasoning (see MindFlow/README.md). All notebooks share one
setup cell: the v4 bundle in Drive MindFlow/ is unpacked once per Colab runtime, the code is read
from MindFlow/code/, and every run is saved under MindFlow/runs/<step>/<RUN_ID>/.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import grid_cells, probe_cells, indicator_cells, daily_cells, daily_pretrained_cells, \
    feature_cells, summary_cells, regime_cells, efficiency_cells, final_cells  # noqa: E402

NOTEBOOKS = {
    "Step1a_Daily_JKX_CNN.ipynb": daily_cells,
    "Step1b_Daily_Pretrained.ipynb": daily_pretrained_cells,
    "Step2a_Intraday_Direction_Models.ipynb": lambda: grid_cells("2a"),
    "Step2b_Intraday_Direction_Pretrained.ipynb": lambda: probe_cells("2b"),
    "Step3a_Intraday_Volatility_Models.ipynb": lambda: grid_cells("3a"),
    "Step3b_Intraday_Volatility_Pretrained.ipynb": lambda: probe_cells("3b"),
    "Step4a_Add_Image_Information.ipynb": indicator_cells,
    "Step4b_Feature_Selection.ipynb": feature_cells,
    "Step5_Summary.ipynb": summary_cells,
    "Step6a_VIX_Regime.ipynb": regime_cells,
    "Step6b_Final_Model.ipynb": final_cells,
    "Step7a_Efficient_Computing.ipynb": efficiency_cells,
}


def write(cells, path, gpu=True):
    for i, c in enumerate(cells):
        c["id"] = f"c{i:02d}"
    meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
            "colab": {"provenance": []}}
    if gpu:
        meta.update(accelerator="GPU")
        meta["colab"]["gpuType"] = "L4"
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5, "metadata": meta}
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out.mkdir(parents=True, exist_ok=True)
    for name, make in NOTEBOOKS.items():
        write(make(), out / name, gpu=not name.startswith(("Step4b", "Step5", "Step6a")))
        print("wrote", out / name)


if __name__ == "__main__":
    main()
