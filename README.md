# Domain Adaptation for Histopathology Image Classification using Cycle Diffusion Model
Official implementation of NIPS 2025 spotlight paper, SGCD: Stain-Guided CycleDiffusion for Unsupervised Domain Adaptation of Histopathology Image Classification

This project utilizes the Cycle Diffusion model for domain adaptation of CAMELYON17 histopathology images. The core idea is to transform the image style of one domain (e.g., scanned by one hospital) to another domain (another hospital's scanning style), while preserving its pathological content (e.g., normal or tumor cells), thereby improving the model's generalization ability across different data sources.

This project also integrates the Vahadane stain normalization algorithm as guidance for the diffusion process, to more precisely control the transformation of staining styles.

## 1. Environment Setup

It is recommended to use `conda` or `venv` to create an independent Python virtual environment to avoid package version conflicts.

```bash
# Create a new environment using conda (recommended)
conda create -n cycle_diffusion python=3.8
conda activate cycle_diffusion
```

### Install Dependencies

You can save the following content as a `requirements.txt` file and then run `pip install -r requirements.txt` to install all necessary libraries.

**`requirements.txt` content:**
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

**Installation command:**
```bash
pip install -r requirements.txt
```

**Note:**
*   The versions of `torch` and `torchvision` should match your CUDA version. Please refer to the [PyTorch official website](https://pytorch.org/get-started/locally/) for installation guides.
*   The `spams` library might be difficult to install. If `pip` installation fails, it is highly recommended to install it using `conda`:
    ```bash
    conda install -c conda-forge spams
    ```

## 2. Data and Pre-trained Model Preparation

### a. Dataset Preparation

This project uses the **CAMELYON17** dataset and requires it to be preprocessed into image patches.

1.  **Data Type**: You need to prepare image patches of `96x96` or `256x256` dimensions (depending on `IMG_SIZE` and `TrainingConfig.image_size` settings in `main.py`), in `.png` or `.jpg` format. Please ensure that the image resolution matches the settings in the code.
2.  **Folder Structure**: Please organize your data according to the following structure. You need to create corresponding folders for different "centers" (domains).
    ```
    <Your data root directory>/
    └── center_X_patches_CL0_RL5_256/       # X is domain ID, e.g., 4 or 5
        ├── training/
        │   ├── training/                   # Training set
        │   │   ├── n_..._patch.png        # 'n' represents normal cell image patches
        │   │   └── t_..._patch.png        # 't' represents tumor cell image patches
        │   └── validation/                 # Validation set
        │       ├── n_..._patch.png
        │       └── t_..._patch.png
        └── testing/                        # Test set
            ├── n_..._patch.png
            └── t_..._patch.png
    ```

3.  **Modify Code Path**:
    Open the `main.py` file, find the `__init__` method of the `CoordDataset` class, and modify the `root_path` variable to your own data root directory path.

    **Original code line (within `CoordDataset` class):**
    ```python
    root_path = '/work/CAMELYON17_temp/center_'+str(domain)+'_patches_CL0_RL5_256/'+folder
    ```
    **Modified example (please replace with your actual path, e.g., `/home/user/my_camelyon_data`):**
    ```python
    root_path = '/path/to/your/camelyon17_data/center_'+str(domain)+'_patches_CL0_RL5_256/'+folder
    ```
    **Note**: The `IMG_SIZE` variable in `main.py` is set to 96, but the folder name contains `256`. Please ensure that your image patch actual dimensions are consistent with the `IMG_SIZE` and `transforms.Resize` settings used in the code.

### b. Pre-trained Model Preparation

The script needs to load pre-trained **classifiers** and **diffusion models** before starting training.

1.  **Required Files and Folders**:
    *   **Classifier Model**: `CAMELYON17_domain_X_resnet.pt` (where `X` is the domain ID, e.g., 4 or 5).
    *   **Diffusion Model**: A folder named `ddpm_CAMELYON17/domain_X/`, which contains subdirectories and files such as `model_index.json`, `scheduler/`, and `unet/`. These are the model storage formats used by the Hugging Face `diffusers` library.

2.  **Storage Location**:
    Please place these pre-trained model files and folders in the **project's root directory**, at the same level as `main.py`. For example:

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
    If your current pre-trained model storage location is different, please modify the paths for loading these models in `main.py` accordingly.

## 3. How to Run the Model

### a. Configure `accelerate`

Before the first run, you need to configure `accelerate` to match your hardware environment (e.g., number of GPUs used, whether mixed precision is enabled). Run the following command in the terminal:

```bash
accelerate config
```

The system will ask a series of questions. Please select according to your actual situation. For a basic setup with a single machine and single GPU, you can refer to the following example answers:

*   `In which compute environment are you running?`: `This machine`
*   `Which type of machine are you using?`: `No distributed training`
*   `Do you want to run your training on CPU?`: `NO`
*   `Do you want to use DeepSpeed?`: `NO`
*   `How many GPUs do you wish to use?`: `1`
*   `Do you wish to use FP16 or BF16 (mixed precision)?`: `no` (or choose `fp16` or `bf16` according to your GPU capabilities and needs)

### b. Adjust Training Parameters

You can directly modify the attributes of the `TrainingConfig` class in `main.py` to adjust training parameters, such as:

*   `SOURCE_DOMAIN`: Set the ID of the source domain (e.g., `5`).
*   `TARGET_DOMAIN`: Set the ID of the target domain (e.g., `4`).
*   `num_epochs`: Total number of training epochs.
*   `learning_rate`: Learning rate.
*   `T`: Timestep for the diffusion process.
*   `guide_until`: Diffusion guidance stops before how many timesteps.
*   `mini_batch`: Number of mini-batches trained per epoch (Note: This is different from `BATCH_SIZE`, which is the number of samples per mini-batch).

### c. Start Training

After completing `accelerate` configuration and all path modifications, in the project root directory, within your activated virtual environment, use the following command to start training:

```bash
accelerate launch main.py
```

## 4. Expected Results

After the model runs successfully, you will see the following outputs:

### a. Terminal Output

*   **Progress Bar**: `tqdm` will display the training progress for each epoch.
*   **Loss Values**: `noise loss` (reconstruction loss), `CE loss` (classification loss), and other metrics will be displayed.
*   **Evaluation Results**: During each test cycle (controlled by `config.test_epochs`), the `Accuracy` and `ROC AUC` of the model on the target test set will be printed.
*   **Model Save Information**: When the model's AUC on the test set reaches a new best value, a message indicating the model save will be displayed.

```
training epoch 0 time step is 30 guide until 10 mode D
100%|██████████| 10/10 [01:30<00:00,  9.00s/it, noise loss: 0.1234；CE loss: 0.6789]
Test Error:
Accuracy: 85.1%
ROC AUC: 0.925
model save at AUC = 0.925
```

### b. Generated Files

After training is complete or during the training process, the following files and folders will be generated in the project root directory:

*   **Diffusion Model Output Directory**:
    *   `ddpm_CAMELYON17/domain_XnY_X/` (e.g., `ddpm_CAMELYON17/domain_5n4_5/`)
    *   `ddpm_CAMELYON17/domain_XnY_Y/` (e.g., `ddpm_CAMELYON17/domain_5n4_4/`)
    *   These directories contain the trained diffusion models (UNet, Scheduler, etc.), which are in the `DDPMPipeline` format and can be used for subsequent inference or generation tasks.
    *   Each directory will also have a `samples/` subdirectory, containing sample images (`recon.png`) generated during training, used for visualizing the effects of diffusion and domain adaptation.

*   **Best Classifier Model**:
    *   `CycelDiffusion_Camelyon17_XnY_0726.pth` (where `XnY` will be replaced by your configured `SOURCE_DOMAIN` and `TARGET_DOMAIN`)
    *   This is the target domain classifier model (its `state_dict`) that completed training and achieved the best AUC on the target test set.

```@inproceedings{
chen2025sgcd,
title={{SGCD}: Stain-Guided CycleDiffusion for Unsupervised Domain Adaptation of Histopathology Image Classification},
author={Hsi-Ling Chen and Chun-Shien Lu and Pau-Choo Chung},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=z2SGaPIhLT}
}
