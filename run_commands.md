# SparseInst Pipeline: Training to TensorRT Deployment

This document provides the standard workflow for training, exporting, and deploying the SparseInst model.

---

## 📂 Environment & Paths
| Component | Local Path |
| :--- | :--- |
| **Project Root** | `/home/aiviz05/Projects/SparseInst` |
| **Config** | `configs/sparse_inst_r50_base.yaml` |
| **Weights** | `output/sparse_inst_r50_base/model_final.pth` |
| **ONNX Model** | `output/sparse_inst_r50_base.onnx` |
| **TRT Engine** | `output/sparinst_engine.trt` |

---

## 🚀 Step 1: Training
Run the training script using the base configuration on a single GPU.

```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_base.yaml \
    --num-gpus 1
```
Ah, I see what's happening. When you copy-paste text that is **already inside** a code block, some editors treat it as plain text instead of rendering it as a formatted document.

To fix this, **copy the text below starting from the first `#` symbol.**

---


## Step 2: Convert PyTorch to ONNX

This step translates your trained model into a universal format. It maps the weights from your `.pth` file into the ONNX graph.

```bash
python onnx/convert_onnx.py \
    --config-file configs/sparse_inst_r50_base.yaml \
    --output output/sparse_inst_r50_base.onnx \
    --opts MODEL.WEIGHTS /home/aiviz05/Projects/SparseInst/output/sparse_inst_r50_base/model_final.pth

```

---

## Step 3: Build TensorRT Engine

Optimize the ONNX file for your specific GPU hardware. This creates the `.trt` engine for high-speed inference.

```bash
python3 /home/aiviz05/Projects/SparseInst/trt/build_engine.py \
    -o /home/aiviz05/Projects/SparseInst/output/sparse_inst_r50_base.onnx \
    -e /home/aiviz05/Projects/SparseInst/output/sparinst_engine.trt \
    -c /home/aiviz05/Projects/SparseInst/output/sparse_inst_r50_base/config.yaml \
    -p fp32 \
    -v 

```

---

## 🔍 Step 4: Run TensorRT Inference

Verify the model performance by running the engine on a sample image.

```bash
python3 trt/infer.py \
    -e /home/aiviz05/Projects/SparseInst/output/sparinst_engine.trt \
    -i "/media/aiviz05/New Volume/Data/TISSMART/Detectron/cvat-benchmark/images/blackpurple-0009.bmp" \
    --det2_config /home/aiviz05/Projects/SparseInst/output/sparse_inst_r50_base/config.yaml \
    -o /home/aiviz05/Projects/SparseInst/output/results_trt

```

---

**Tip:** If you are using an editor like VS Code or Obsidian, make sure the file extension ends in **.md** (e.g., `README.md`) for the formatting to appear.

Would you like me to generate a single shell script file so you can run all three steps with one command?