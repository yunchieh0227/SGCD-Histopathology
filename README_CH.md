# 使用循環擴散模型進行組織病理學圖像領域適應
Official implementation of NIPS 2025 spotlight paper, SGCD: Stain-Guided CycleDiffusion for Unsupervised Domain Adaptation of Histopathology Image Classification

在處理影像型的無監督領域適應（Unsupervised Domain Adaptation, UDA）問題時，領域轉換（domain translation）的效果取決於生成影像的品質，以及是否能保留關鍵的判別特徵。然而，要實現高品質且穩定的影像轉換，通常需要成對（paired）的資料，這在目標領域標註有限的情況下會造成困難。

為了解決此問題，本文提出一種新方法，稱為 Stain-Guided Cycle Diffusion（SGCD）。該方法採用一個具有雙向生成約束的雙擴散模型（dual diffusion model），以合成高度真實的資料，用於下游任務的微調（fine-tuning）。雙向生成約束能夠在控制生成過程的同時，確保轉換後的影像保留對下游模型至關重要的特徵。

此外，本文還引入了一種染色引導一致性損失（stain-guided consistency loss），以增強雙擴散模型的去噪能力，從而提升跨領域影像轉換的品質。該方法能夠使用來自一個領域的潛在表示（latents），搭配在另一個領域上訓練的擴散模型，實現高品質的影像轉換。

在四個公開資料集上的實驗結果顯示，SGCD 能有效提升目標領域中下游任務模型的性能。

## 1. 環境建置

建議使用 `conda` 或 `venv` 建立一個獨立的 Python 虛擬環境，以避免套件版本衝突。

```bash
# 使用 conda 建立新環境 (推薦)
conda create -n cycle_diffusion python=3.8
conda activate cycle_diffusion
```

### 安裝依賴套件

您可以將以下內容儲存為 `requirements.txt` 檔案，然後執行 `pip install -r requirements.txt` 來安裝所有必要的函式庫。

**`requirements.txt` 內容:**
```
torch>=1.10.0
torchvision>=0.11.0
diffusers>=0.10.0
accelerate>=0.10.0
scikit-learn>=1.0.0
numpy>=1.20.0
opencv-python>=4.0.0
matplotlib>=3.0.0
tqdm>=4.0.0
Pillow>=9.0.0
spams
```

**安裝指令:**
```bash
pip install -r requirements.txt
```

**注意：**
*   `torch` 和 `torchvision` 的版本應與您的 CUDA 版本相匹配。請參考 [PyTorch 官網](https://pytorch.org/get-started/locally/) 的安裝指南。
*   `spams` 函式庫可能較難安裝。如果 `pip` 安裝失敗，強烈建議使用 `conda` 來安裝：
    ```bash
    conda install -c conda-forge spams
    ```

## 2. 資料與預訓練模型準備

### a. 資料集準備

本專案使用 **CAMELYON17** 資料集，並且需要將其預處理為圖像塊（patches）。

1.  **資料類型**：您需要準備好 `96x96` 或 `256x256` 尺寸的圖像塊（根據 `main.py` 中的 `IMG_SIZE` 和 `TrainingConfig.image_size` 設定而定），檔案格式為 `.png` 或 `.jpg`。請確保圖像解析度與程式碼中的設定一致。
2.  **資料夾結構**：請依照以下結構組織您的資料。您需要為不同的「中心」（domain）建立對應的資料夾。
    ```
    <您的資料根目錄>/
    └── center_X_patches_CL0_RL5_256/       # X 是 domain ID, 例如 4 或 5
        ├── training/
        │   ├── training/                   # 訓練集
        │   │   ├── n_..._patch.png        # 'n' 代表正常細胞圖像塊
        │   │   └── t_..._patch.png        # 't' 代表腫瘤細胞圖像塊
        │   └── validation/                 # 驗證集
        │       ├── n_..._patch.png
        │       └── t_..._patch.png
        └── testing/                        # 測試集
            ├── n_..._patch.png
            └── t_..._patch.png
    ```

3.  **修改程式碼路徑**：
    打開 `main.py` 檔案，找到 `CoordDataset` 類別的 `__init__` 方法，並將 `root_path` 變數修改為您自己的資料根目錄路徑。

    **原始程式碼行 (在 `CoordDataset` 類別內):**
    ```python
    root_path = '/work/CAMELYON17_temp/center_'+str(domain)+'_patches_CL0_RL5_256/'+folder
    ```
    **修改後範例 (請替換為您的實際路徑，例如 `/home/user/my_camelyon_data`):**
    ```python
    root_path = '/path/to/your/camelyon17_data/center_'+str(domain)+'_patches_CL0_RL5_256/'+folder
    ```
    **注意:** `main.py` 中 `IMG_SIZE` 變數設定為 96，但資料夾名稱包含 `256`。請確保您的圖像塊實際尺寸與程式碼中使用的 `IMG_SIZE` 和 `transforms.Resize` 設定一致。

### b. 預訓練模型準備

腳本在開始訓練前需要載入預訓練好的**分類器**和**擴散模型**。

1.  **所需檔案和資料夾**：
    *   **分類器模型**：`CAMELYON17_domain_X_resnet.pt` (其中 `X` 是 domain ID，例如 4 或 5)。
    *   **擴散模型**：一個名為 `ddpm_CAMELYON17/domain_X/` 的資料夾，其中包含 `model_index.json`, `scheduler/` 和 `unet/` 等子目錄和檔案。這些是 Hugging Face `diffusers` 庫所使用的模型儲存格式。

2.  **存放位置**：
    請將這些預訓練模型檔案和資料夾放置在**專案的根目錄**下，與 `main.py` 位於同一層級。例如：

    ```
    your_project_root/
    ├── main.py
    ├── resize_method.py
    ├── vahadane.py
    ├── CAMELYON17_domain_4_resnet.pt
    ├── CAMELYON17_domain_5_resnet.pt
    └── ddpm_CAMELYON17/
        ├── domain_4/
        │   ├── model_index.json
        │   ├── scheduler/
        │   └── unet/
        └── domain_5/
            ├── model_index.json
            ├── scheduler/
            └── unet/
    ```
    如果您目前的預訓練模型存放位置不同，請相應修改 `main.py` 中載入這些模型的路徑。

## 3. 如何運行模型

### a. 配置 `accelerate`

在第一次運行前，您需要配置 `accelerate` 以匹配您的硬體環境（例如使用的 GPU 數量、是否啟用混合精度等）。在終端機中執行以下指令：

```bash
accelerate config
```

系統會提出一系列問題。請根據您的實際情況進行選擇。對於單機單 GPU 的基本設定，您可以參考以下範例回答：

*   `In which compute environment are you running?`: `This machine`
*   `Which type of machine are you using?`: `No distributed training`
*   `Do you want to run your training on CPU?`: `NO`
*   `Do you want to use DeepSpeed?`: `NO`
*   `How many GPUs do you wish to use?`: `1`
*   `Do you wish to use FP16 or BF16 (mixed precision)?`: `no` (或根據您的 GPU 能力和需求選擇 `fp16` 或 `bf16`)

### b. 調整訓練參數

您可以直接在 `main.py` 中修改 `TrainingConfig` 類別的屬性來調整訓練參數，例如：

*   `SOURCE_DOMAIN`：設定源領域的 ID (例如 `5`)。
*   `TARGET_DOMAIN`：設定目標領域的 ID (例如 `4`)。
*   `num_epochs`：訓練的總輪數。
*   `learning_rate`：學習率。
*   `T`：擴散過程的時間步長。
*   `guide_until`：擴散引導在多少時間步之前停止。
*   `mini_batch`：每次 epoch 訓練的 mini-batch 數量 (注意：這與 `BATCH_SIZE` 不同，`BATCH_SIZE` 是每個 mini-batch 的樣本數)。

### c. 啟動訓練

完成 `accelerate` 配置和所有路徑修改後，請在專案根目錄下，於您啟用的虛擬環境中，使用以下指令來啟動訓練：

```bash
accelerate launch main.py
```

## 4. 預期結果

模型成功運行後，您會看到以下產出：

### a. 終端機輸出

*   **進度條**：`tqdm` 會顯示每個 epoch 的訓練進度。
*   **損失值**：會顯示 `noise loss` (重建損失)、`CE loss` (分類損失) 等指標。
*   **評估結果**：在每個測試週期（由 `config.test_epochs` 控制），會打印出模型在目標測試集上的 `Accuracy` (準確率) 和 `ROC AUC`。
*   **模型保存信息**：當模型在測試集上的 AUC 達到新的最佳值時，會顯示保存模型的訊息。

```
training epoch 0 time step is 30 guide until 10 mode D
100%|██████████| 10/10 [01:30<00:00,  9.00s/it, noise loss: 0.1234；CE loss: 0.6789]
Test Error:
Accuracy: 85.1%
ROC AUC: 0.925
model save at AUC = 0.925
```

### b. 生成的檔案

訓練完成或在訓練過程中，會在專案根目錄下生成以下檔案和資料夾：

*   **擴散模型輸出目錄**：
    *   `ddpm_CAMELYON17/domain_XnY_X/` (例如 `ddpm_CAMELYON17/domain_5n4_5/`)
    *   `ddpm_CAMELYON17/domain_XnY_Y/` (例如 `ddpm_CAMELYON17/domain_5n4_4/`)
    *   這些目錄中包含訓練好的擴散模型（UNet、Scheduler 等），它們是 `DDPMPipeline` 的格式，可用於後續的推理或生成任務。
    *   每個目錄內還會有一個 `samples/` 子資料夾，其中包含訓練過程中生成的樣本圖像 (`recon.png`)，用於視覺化擴散和領域適應的效果。

*   **最佳分類器模型**：
    *   `CycelDiffusion_Camelyon17_XnY_0726.pth` (其中 `XnY` 會替換為您設定的 `SOURCE_DOMAIN` 和 `TARGET_DOMAIN`)
    *   這是訓練完成且在目標測試集上達到最佳 AUC 的目標領域分類器模型（其 `state_dict`）。

```bibtex
@inproceedings{
chen2025sgcd,
title={{SGCD}: Stain-Guided CycleDiffusion for Unsupervised Domain Adaptation of Histopathology Image Classification},
author={Hsi-Ling Chen and Chun-Shien Lu and Pau-Choo Chung},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=z2SGaPIhLT}
}
```
