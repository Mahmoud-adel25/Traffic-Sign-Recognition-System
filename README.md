# 🚦 Traffic Sign Recognition System

A comprehensive deep learning system for traffic sign recognition using the GTSRB (German Traffic Sign Recognition Benchmark) dataset.

## 🌟 Live Demo

**Try the live application:** [Traffic Sign Recognition App](https://mahmoud-adel25-traffic-sign-recognition-system-appapp-lfhfoh.streamlit.app/)

> 🚀 **Experience real-time traffic sign classification with our deployed model!** Upload your traffic sign images and get instant predictions with confidence scores.

## 📋 Project Overview

This project implements a complete traffic sign recognition system using deep learning techniques. It includes data preprocessing, model training, evaluation, and a web application for real-time predictions.

### 🎯 Features

- **🌐 Live Web Application**: Deployed on Streamlit Cloud for instant access
- **🎯 Multi-class Classification**: Recognizes 43 different traffic sign classes
- **🤖 Two Model Architectures**: Custom CNN and MobileNetV2 transfer learning
- **📈 Data Augmentation**: Improves model robustness and performance
- **📊 Comprehensive Evaluation**: Accuracy metrics, confusion matrices, and detailed analysis
- **⚡ Real-time Predictions**: Upload images and get instant classification results
- **🔄 Model Comparison**: Side-by-side performance analysis
- **🎨 Modern UI**: Dark theme with responsive design and interactive visualizations

## 🏗️ Project Structure

```
Traffic Sign Recognition Description/
├── Data/
│   └── Dataset/
│       ├── Train/          # Training images organized by class
│       ├── Test/           # Test images
│       ├── Meta/           # Class metadata and sample images
│       ├── Train.csv       # Training data labels
│       └── Test.csv        # Test data labels
├── Src/
│   ├── train.py           # Main training script
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   ├── model_evaluation.py    # Model evaluation and analysis
│   └── utils.py           # Helper functions
├── App/
│   ├── app.py             # Streamlit web application
│   └── prediction.py      # Prediction utilities
├── models/                # Trained model files
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## 🚀 Quick Start

### Option 1: Use Live Demo (Recommended)
**🌐 No installation required!** Simply visit the [live application](https://mahmoud-adel25-traffic-sign-recognition-system-appapp-lfhfoh.streamlit.app/) and start classifying traffic signs immediately.

### Option 2: Local Installation

#### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Traffic-Sign-Recognition-Description

# Install dependencies
pip install -r requirements.txt
```

#### 2. Training Models

```bash
# Train both CNN and MobileNetV2 models
python Src/train.py
```

#### 3. Run Web Application

```bash
# Start the Streamlit app locally
streamlit run App/app.py
```

## 📊 Model Performance

### Custom CNN Model
- **Test Accuracy**: 96%
- **Training Time**: ~10 minutes
- **Model Size**: 13MB
- **Best Performance**: Excellent for real-time applications

### MobileNetV2 Model
- **Test Accuracy**: 53%
- **Training Time**: ~8 minutes
- **Model Size**: 11MB
- **Note**: Underperformed due to small input size (48x48)

## 🔧 Technical Details

### Data Preprocessing
- **Image Resizing**: 48x48 pixels
- **Normalization**: Pixel values scaled to [0,1]
- **Data Augmentation**: Rotation, width/height shift, zoom
- **Color Conversion**: BGR to RGB

### Model Architectures

#### Custom CNN
```python
- Conv2D(32, 3x3) + ReLU
- MaxPooling2D(2x2)
- Conv2D(64, 3x3) + ReLU
- MaxPooling2D(2x2)
- Conv2D(128, 3x3) + ReLU
- Flatten()
- Dense(128) + ReLU + Dropout(0.5)
- Dense(43) + Softmax
```

#### MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet
- Global Average Pooling
- Dense(128) + ReLU + Dropout(0.3)
- Dense(43) + Softmax

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions

## 🎨 Web Application Features

### 🌐 Live Demo Highlights
- **🚀 Instant Access**: No installation required - use directly in your browser
- **📱 Mobile Friendly**: Responsive design works on all devices
- **⚡ Real-time Prediction**: Upload and classify traffic signs instantly
- **🎛️ Model Selection**: Choose between Custom CNN and MobileNetV2 with radio buttons
- **📊 Confidence Scores**: View prediction confidence levels with color-coded indicators
- **🏆 Top-5 Predictions**: See multiple possible classifications with rankings
- **📈 Interactive Visualizations**: Charts and graphs with dark theme
- **🎨 Modern UI**: Beautiful dark theme with glassmorphism effects
- **📋 Detailed Analysis**: Comprehensive performance metrics and confusion matrices
- **📚 Reference Guide**: Complete traffic sign class information

## 📈 Results Analysis

### Custom CNN Performance
- Achieved 96% accuracy on test set
- Excellent performance across most traffic sign classes
- Some challenges with rare classes (class 27, 29, 30)
- Robust to variations in lighting and orientation

### MobileNetV2 Performance
- Lower accuracy (53%) due to input size constraints
- Better suited for larger input sizes (224x224)
- Transfer learning benefits limited by dataset size

## 🔍 Key Insights

1. **Data Quality**: GTSRB dataset provides high-quality, well-labeled images
2. **Model Architecture**: Custom CNN outperformed MobileNetV2 for this specific task
3. **Data Augmentation**: Significantly improved model generalization
4. **Class Imbalance**: Some classes have fewer samples, affecting performance
5. **Real-world Applicability**: Custom CNN suitable for deployment

## 🛠️ Technologies Used

### 🧠 Machine Learning & AI
- **Python 3.12**
- **TensorFlow 2.19.0**
- **OpenCV 4.8.1**
- **Scikit-learn** (Evaluation Metrics)

### 🌐 Web Application & Deployment
- **Streamlit** (Web Interface & Cloud Deployment)
- **Plotly** (Interactive Visualizations)
- **Streamlit Cloud** (Hosting Platform)

### 📊 Data Processing & Visualization
- **Pandas & NumPy** (Data Processing)
- **Matplotlib & Seaborn** (Static Visualization)
- **Plotly Express** (Interactive Charts)

## 📝 Requirements Covered

✅ **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark)  
✅ **Deep Learning**: CNN and MobileNetV2 models  
✅ **Image Preprocessing**: Resizing, normalization, augmentation  
✅ **Multi-class Classification**: 43 traffic sign classes  
✅ **Performance Evaluation**: Accuracy, confusion matrix, detailed metrics  
✅ **Data Augmentation**: Rotation, shifts, zoom for improved performance  
✅ **Model Comparison**: Custom CNN vs pre-trained MobileNetV2  
✅ **Web Application**: Real-time prediction interface  
✅ **Cloud Deployment**: Live demo on Streamlit Cloud  
✅ **Interactive UI**: Modern dark theme with responsive design  

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GTSRB dataset creators
- TensorFlow and Keras communities
- Streamlit for the web framework
- OpenCV for computer vision capabilities
