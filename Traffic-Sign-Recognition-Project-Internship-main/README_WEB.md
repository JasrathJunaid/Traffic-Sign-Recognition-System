# 🚦 Traffic Sign Recognition Web App

## Real-Time Few-Shot Learning Web Application

A modern, interactive web application for traffic sign recognition using few-shot learning. Built with Streamlit for an intuitive user experience.

## 🌟 Features

### 📸 **Upload Image**
- Drag & drop or click to upload traffic sign images
- Supports PNG, JPG, JPEG formats
- Instant prediction with confidence scores

### 📹 **Real-Time Camera**
- Live camera capture from webcam
- Auto-predict on capture
- Real-time traffic sign detection

### 📊 **Visual Results**
- Input image display
- Prediction with confidence meter
- Probability distribution chart
- Color-coded confidence levels

### ℹ️ **About Section**
- Model information and architecture
- Performance metrics
- Dataset details
- Technical specifications

## 🚀 Quick Start

### Option 1: Automatic Setup
```bash
python run_app.py
```
This will install Streamlit and launch the app automatically.

### Option 2: Manual Setup
```bash
# Install dependencies
pip install streamlit

# Run the app
streamlit run app.py
```

The app will open in your browser at: **http://localhost:8501**

## 🎯 How to Use

### 1. **Upload Image Tab**
1. Click "Browse files" or drag & drop an image
2. Click "🔮 Predict" button
3. View results with confidence scores and probability charts

### 2. **Camera Capture Tab**
1. Allow camera access in your browser
2. Position traffic sign in camera view
3. Click "📹 Take photo" or enable auto-predict
4. Get instant results

### 3. **About Tab**
- Learn about few-shot learning
- View model architecture
- Check performance metrics
- Understand the dataset

## 🧠 Model Details

### Architecture
- **Base Model**: Simple CNN with 4 convolutional layers
- **Input**: 84x84 RGB images
- **Output**: 5 traffic sign classes
- **Training**: Few-shot learning with 6-11 samples per class

### Classes
1. **No Entry** (11 samples)
2. **No Parking** (9 samples)
3. **Pedestrian Crossing** (8 samples)
4. **Speed Limit** (10 samples)
5. **Stop Sign** (6 samples)

### Performance
- **Inference Time**: < 1 second
- **Accuracy**: ~85% (with trained model)
- **Confidence Scoring**: Real-time probability distribution

## 🎨 User Interface

### Design Features
- **Modern Layout**: Clean, responsive design
- **Color Coding**: 
  - 🟢 High confidence (>80%)
  - 🟡 Medium confidence (60-80%)
  - 🔴 Low confidence (<60%)
- **Interactive Charts**: Probability distributions
- **Progress Bars**: Visual confidence indicators

### Navigation
- **Tabbed Interface**: Easy switching between features
- **Sidebar Information**: Model details and classes
- **Responsive Design**: Works on desktop and mobile

## 🔧 Technical Stack

### Frontend
- **Streamlit**: Python web framework
- **HTML/CSS**: Custom styling
- **JavaScript**: Camera integration

### Backend
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **PIL**: Image manipulation
- **NumPy**: Numerical operations

### Model
- **CNN Architecture**: 4-layer convolutional network
- **Few-Shot Learning**: Prototypical networks
- **Data Augmentation**: Random flips, color jittering

## 📱 Browser Compatibility

### Supported Browsers
- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

### Required Features
- **WebRTC**: For camera access
- **File API**: For image uploads
- **Canvas API**: For image processing

## 🐛 Troubleshooting

### Common Issues

#### Camera Not Working
1. Check browser camera permissions
2. Ensure HTTPS connection (localhost is fine)
3. Try refreshing the page

#### Model Not Loading
1. Ensure `simple_cnn.pth` exists
2. Run `python train.py` first
3. Check file permissions

#### Slow Performance
1. Close other browser tabs
2. Ensure sufficient RAM
3. Try smaller images

#### Installation Issues
```bash
# Update pip
python -m pip install --upgrade pip

# Install streamlit separately
pip install streamlit

# Install all dependencies
pip install -r web_requirements.txt
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Cloud Deployment
```bash
# Streamlit Cloud
# Deploy directly from GitHub repository

# Heroku
# Add Procfile and requirements.txt

# Docker
# Build container with Streamlit
```

### Custom Port
```bash
streamlit run app.py --server.port 8080
```

## 📊 Monitoring

### Performance Metrics
- **Response Time**: Track prediction speed
- **Accuracy**: Monitor model performance
- **Usage Analytics**: User interaction data

### Logs
```bash
# Enable debug mode
streamlit run app.py --logger.level debug

# View logs
tail -f streamlit_logs.txt
```

## 🔒 Security

### Data Privacy
- **No Data Storage**: Images processed locally
- **No Cloud Upload**: All processing on-device
- **Temporary Files**: Auto-cleanup after processing

### Best Practices
- **Input Validation**: Check file types and sizes
- **Error Handling**: Graceful failure messages
- **Resource Limits**: Prevent memory issues

## 🎯 Future Enhancements

### Planned Features
- [ ] **Video Support**: Real-time video stream analysis
- [ ] **Batch Processing**: Multiple image upload
- [ ] **Model Training**: Train new models in the app
- [ ] **Export Results**: Download prediction reports
- [ ] **Mobile App**: React Native version

### Advanced Features
- [ ] **Custom Classes**: Add new traffic sign types
- [ ] **Model Comparison**: Compare different architectures
- [ ] **Performance Analytics**: Detailed usage statistics
- [ ] **API Integration**: REST API for developers

## 📞 Support

### Getting Help
1. Check the troubleshooting section
2. Review the documentation
3. Open an issue on GitHub
4. Contact the development team

### Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a pull request
4. Join the community

---

**Built with ❤️ using Streamlit and PyTorch**

*Traffic Sign Recognition - Making roads safer with AI*
