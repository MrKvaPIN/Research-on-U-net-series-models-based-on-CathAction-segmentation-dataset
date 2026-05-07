# CathAction U-Net Variant Segmentation

This project implements and compares several U-Net-based deep learning models for catheter and guidewire segmentation in endovascular X-ray images from the CathAction dataset. The task is formulated as a three-class semantic segmentation problem, including background, catheter, and guidewire regions.

The project includes four segmentation models: U-Net with a ResNet34 encoder, R2U-Net, Attention U-Net, and Attention R2U-Net. Each model is trained under a consistent experimental pipeline using PyTorch, OpenCV, Albumentations, and NumPy-based mask annotations. Images are resized to 768 × 768, and the models are evaluated using mean Dice score and mean IoU.

To address class imbalance caused by the small foreground regions of guidewires and catheters, a hybrid loss function combining multi-class Dice loss and focal loss is used. The training pipeline also includes data augmentation, validation after each epoch, learning-rate scheduling, and automatic checkpoint saving based on the best validation Dice score.

Experimental results show that the standard U-Net and Attention U-Net achieved more stable validation performance than the recurrent residual variants under the current dataset and training configuration. This project provides a reproducible baseline for evaluating U-Net-style architectures on medical X-ray segmentation tasks.
