# IMAGE-CLASSIFICATION-MODEL

"COMPANY": CODTECH IT SOLLUTIONS

"NAME": T.SRIYANSH

"INTERN ID":CT04DM652

"DOMAIN": MACHINE LEARNING

"DURATION": 4 WEEKS

"MENTOR": NEELA SANTHOSH



 DESCRIPTION

### **Convolutional Neural Network for Image Classification: Intel Image Dataset**  

Image classification is a fundamental task in computer vision, where a model learns to categorize images into predefined classes. In this project, we build a **Convolutional Neural Network (CNN)** using **TensorFlow** to classify natural scene images from the **Intel Image Classification Dataset**, available on Kaggle. This dataset contains **high-resolution (150x150) images** divided into six categories: buildings, forests, glaciers, mountains, seas, and streets. Unlike low-resolution datasets like CIFAR-10, these images provide clearer visual details, making them more suitable for real-world applications.  

 **Dataset & Preprocessing**  
The dataset is structured into training and testing folders, each containing subdirectories for the six classes. We use **ImageDataGenerator** from TensorFlow to efficiently load and preprocess the images. To improve model generalization, we apply **data augmentation** techniques such as random rotations, shifts, shearing, zooming, and horizontal flipping. This artificially expands the training dataset, helping the model learn robust features without overfitting. The pixel values are normalized (rescaled to 0-1) to ensure faster and more stable training.  

 **CNN Architecture**  
Our CNN consists of:  
1. **Convolutional Layers:** These layers extract hierarchical features from images using filters. We use four blocks of Conv2D layers with increasing filters (32, 64, 128, 256) and ReLU activation to introduce non-linearity.  
2. **MaxPooling Layers:** After each convolutional layer, we apply max-pooling to reduce spatial dimensions, retaining the most important features while reducing computation.  
3. **Dropout Layer:** A dropout rate of 0.5 is applied to prevent overfitting by randomly deactivating neurons during training.  
4. **Dense Layers:** The flattened features are passed through a fully connected layer (512 neurons) before the final softmax layer, which outputs class probabilities.  

 **Training & Evaluation**  
The model is trained using the **Adam optimizer** and **categorical cross-entropy loss** for 15 epochs. We monitor **training and validation accuracy/loss** to detect overfitting. After training, we evaluate the model on the test set and generate:  
- **Confusion Matrix:** Shows correct and incorrect predictions per class.  
- **Classification Report:** Provides precision, recall, and F1-score metrics.  
- **Prediction Visualizations:** Displays sample test images with predicted vs. true labels (correct predictions in green, incorrect in red).  

**Expected Results**  
With this architecture, we achieve:  
- **Training Accuracy:** ~90-95%  
- **Validation Accuracy:** ~85-90%  
- **Clear Visualizations:** Unlike low-resolution datasets, the 150x150 images allow better interpretation of model predictions.  

 **Applications & Improvements**  
This model can be deployed in **environmental monitoring, tourism recommendation systems, or satellite image analysis**. For better performance, we could:  
- Use **transfer learning** (e.g., ResNet, EfficientNet).  
- Implement **learning rate scheduling**.  
- Increase model depth or tuning hyperparameters.  

This project demonstrates how CNNs can effectively classify high-resolution images while providing interpretable results through visualizations and performance metrics.
