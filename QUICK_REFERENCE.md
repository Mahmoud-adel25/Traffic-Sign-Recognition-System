# 🚦 Quick Reference Card

## ⚡ Fast Start Guide

### 1. Upload Image
- Click "Browse files" or drag & drop
- Supported: JPG, PNG, JPEG
- Max size: 10MB

### 2. Choose Model
- **Custom CNN** (96% accuracy) - Best quality
- **MobileNetV2** (53% accuracy) - Faster processing

### 3. Adjust Confidence
- **Low (0-30%)**: See all predictions
- **Medium (30-70%)**: Balanced (recommended)
- **High (70-100%)**: Only very confident

### 4. View Results
- Main prediction with confidence
- Top-5 predictions
- Confidence distribution chart

## 🎯 Best Practices

### ✅ Do's
- Use clear, well-lit images
- Focus on one sign per image
- Capture from front angle
- Use high-resolution photos

### ❌ Don'ts
- Upload blurry images
- Include multiple signs
- Use extreme angles
- Upload very large files

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| Low confidence | Try clearer image or lower threshold |
| Wrong prediction | Use Custom CNN model |
| Slow processing | Use MobileNetV2 or smaller image |
| Model error | Check if models are downloaded |

## 📊 Understanding Results

### Confidence Levels
- 🟢 **High (80-100%)**: Very reliable
- 🟡 **Medium (50-80%)**: Good reliability  
- 🔴 **Low (0-50%)**: Uncertain prediction

### Top-5 Predictions
- 🥇 **1st**: Most likely
- 🥈 **2nd**: Second choice
- 🥉 **3rd**: Third option
- **4th & 5th**: Additional possibilities

## 🎛️ Quick Settings

### Model Selection
```
Custom CNN: High accuracy, slower
MobileNetV2: Lower accuracy, faster
```

### Confidence Threshold
```
0.0-0.3: Show all predictions
0.3-0.7: Balanced (default)
0.7-1.0: Only confident predictions
```

## 📱 Mobile Tips

- Use MobileNetV2 for faster processing
- Keep images under 5MB
- Ensure good lighting
- Hold phone steady

## 🆘 Need Help?

1. Check "How to Use This Application" in the app
2. Read the full USER_GUIDE.md
3. Try different images and settings
4. Review model information in sidebar

---

**Remember**: Clear images = Better predictions! 📸✨
