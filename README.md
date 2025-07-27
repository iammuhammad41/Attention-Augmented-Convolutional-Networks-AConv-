# Attention-Augmented-Convolutional-Networks-AConv-
Convolution + Attention: This combines the strengths of both convolution and attention mechanisms, improving the network's ability to focus on relevant image regions while maintaining the computational efficiency of convolutions.

# Muffin vs. Chihuahua Image Classification

A PyTorch Lightning pipeline for binary image classification to distinguish muffins from chihuahuas. This Colab-compatible notebook uses pretrained models, Albumentations for augmentation, TorchMetrics for evaluation, and Weights & Biases for experiment tracking.


## 🔗 Project Structure

```
muffin-chihuahua-classification/
├── data/                             # Local or Colab data folders
│   ├── train/                        # Subfolders 'muffin/' and 'chihuahua/' with images
│   └── test/                         # Subfolders 'muffin/' and 'chihuahua/' with images
├── notebook.ipynb                    # Main Colab notebook (code above)
└── README.md                         # This file
```



## 🛠️ Dependencies

Install required packages:

```bash
pip install pretrainedmodels albumentations torch torchvision pytorch-lightning torchmetrics wandb colorama tqdm seaborn matplotlib
```

> **Note:** In Colab, uncomment the pip install line and run directly.



## 🖥️ Environment

* Python 3.7+
* GPU recommended for faster training
* Colab support: adjust paths under `/content/` accordingly



## ⚙️ Configuration

The notebook sets global variables and imports:

* `CONFIG['competetion'] = 'Muffin_Chihuaha'` for W\&B
* Custom color scheme and Seaborn palette

Paths are defined in `build_metadata()` calls:

```python
train_dir = "/content/input/muffin-vs-chihuahua-image-classification/train"
test_dir  = "/content/input/muffin-vs-chihuahua-image-classification/test"
```



## 🚀 Usage

1. **Build metadata**: Scan train/test folders and create shuffled DataFrames of image paths and labels.
2. **Prepare DataModule**: `ImageDataLoader` handles resizing, normalization, and batching via Albumentations.
3. **Initialize model**: Custom CNN with spatial attention and adaptive pooling in `Model` class.
4. **Train**: Use `pl.Trainer` with early stopping and mixed-precision:

   ```python
   trainer = pl.Trainer(
       devices=1, accelerator='gpu', precision=16,
       callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
       max_epochs=100
   )
   trainer.fit(model, train_dl, val_dl)
   ```
5. **Evaluate**: Load best checkpoint and compute test accuracy.
6. **Predict**: Use `custom_img_pred()` to infer on new images.



## 📊 Results

* **Test Accuracy**: \~90.2%
* **Metrics**: Training/validation loss and AUC are logged per epoch.
* **Visualization**: Class distribution and training curves displayed inline.

