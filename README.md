# Source Code for **“Towards Open Environment Intent Prediction”** with **MindSpore**

This repository provides the **MindSpore implementation** of the framework described in the paper *Towards Open Environment Intent Prediction*.
All modules including Prefix-Tuning, Adaptive Decision Boundary (ADB), and OOD class generation have been adapted to support MindSpore execution on **Ascend** or **GPU**.

---

## 📦 Dependencies

### 1. Create Conda Environment

```bash
conda create --name env python=3.7
conda activate env
```

### 2. Install MindSpore

Please install the appropriate MindSpore version depending on your hardware:

**Ascend:**

```bash
pip install mindspore -f https://www.mindspore.cn/whl/ascend910/
```

**GPU:**

```bash
pip install mindspore-gpu
```

### 3. Install additional Python libraries

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run experiments (example: T5_prefix_tuning_with_multi_label_v2) with **MindSpore**

```bash
# 1. Prefix-tuning based multi-label training
python prefix_trainer.py \
    --dataset banking \
    --lr 2e-4 \
    --pre_seq_len 256 \
    --seed 42 \
    --device_id 0

# 2. Adaptive Decision Boundary (ADB) model
python adb_model.py \
    --dataset banking \
    --lr 2e-4 \
    --pre_seq_len 256 \
    --seed 42 \
    --device_id 0

# 3. Generate OOD classes using RAMA-based module
python generate_ood_class.py \
    --dataset banking \
    --lr 2e-4 \
    --pre_seq_len 256 \
    --seed 42 \
    --p_node 0.2 \
    --device_id 0
```

> 注：原 PyTorch 程序中的 `--gpu_id` 参数已经统一替换为更通用的 `--device_id`，以适应 MindSpore 的设备管理方式。

---

## 📊 Results

Experimental results will be automatically saved under the **outputs/** directory.
Example:

```
outputs/banking_adb_result.csv
```

---

## 🙏 Thanks & Acknowledgments

The MindSpore version of this project is adapted based on the following excellent works:

* **Adaptive-Decision-Boundary (ADB)**
  [https://github.com/hanleizhang/Adaptive-Decision-Boundary](https://github.com/hanleizhang/Adaptive-Decision-Boundary)

* **PrefixTuning**
  [https://github.com/XiangLi1999/PrefixTuning](https://github.com/XiangLi1999/PrefixTuning)

* **RAMA**
  [https://github.com/pawelswoboda/RAMA](https://github.com/pawelswoboda/RAMA)
