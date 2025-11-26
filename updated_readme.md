# 👁️ RetinaCare: Fusion DR Predictor (MVP)

## 🔗 Live Demo
**🌐 [Try the Live Model on Hugging Face Spaces](https://huggingface.co/spaces/MudLegacy/retinacare-dr-classifier)**

**🌐 [Full Web Application](https://retinacare.com)** *(Coming Soon - Under Development)*

---

## 📖 Project Overview
**RetinaCare Fusion DR Predictor** is an AI-powered diagnostic tool designed to **predict the severity of Diabetic Retinopathy (DR)** by fusing retinal image features with patient clinical data.  

This system supports **early detection**, **risk assessment**, and **personalized follow-up recommendations**, empowering healthcare providers to make informed decisions. The ML model has been deployed both as a **standalone demo on Hugging Face** and integrated into a **full-stack web application** developed by the RetinaCare team for production use.

### ✨ Key Features
- 🔍 **Multimodal AI Analysis** - Combines retinal images with clinical data
- 📊 **5-Class DR Severity Classification** - From No DR to Proliferative DR
- 💡 **Intelligent Risk Assessment** - Evidence-based follow-up recommendations
- 🌐 **Dual Deployment** - Hugging Face demo + Full-stack web application
- 🔌 **Backend Integration** - Model API for seamless healthcare system integration
- ⚡ **Real-Time Predictions** - Instant analysis and results

---

## 🗺️ Model Development Workflow

### 1. Problem Definition
The model classifies Diabetic Retinopathy (DR) severity into five categories:
- **No DR** - No visible signs of diabetic retinopathy
- **Mild NPDR** - Presence of microaneurysms only
- **Moderate NPDR** - More than just microaneurysms, but less than severe NPDR
- **Severe NPDR** - Extensive hemorrhages, venous beading, or IRMA
- **Proliferative DR (PDR)** - Growth of new blood vessels (highest risk of vision loss)

Additionally, the system provides **follow-up recommendations** based on predicted risk levels.

---

### 2. Data Sources
- **Image Data:** Retinal fundus images (preprocessed and resized to 224×224)  
- **Clinical Data:** Three key patient features:
  - HbA1c level (%)
  - Systolic Blood Pressure (mmHg)
  - Duration of Diabetes (years)

📂 **Dataset Used:** [EyePACS, APTOS & Messidor Diabetic Retinopathy Dataset](https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy)

This combined dataset provides a diverse and comprehensive collection of retinal images suitable for building and validating DR severity models.

---

### 3. Image Feature Extraction
A **pretrained ResNet50** model from `torchvision` was utilized to extract deep image features:
- Uses pretrained ImageNet weights for transfer learning
- The final classification layer is removed to extract feature vectors
- The resulting feature vector per image is **2048-dimensional**
- Features are frozen and used as input to the fusion model

This approach leverages **transfer learning**, allowing the model to capture rich visual representations without retraining the full CNN.

---

### 4. Fusion Model Architecture
The model fuses **image** and **clinical** features into a single prediction pipeline:

```
[Retinal Image 224×224]
        ↓
   ResNet50 (frozen)
        ↓
[Image Features: 2048-d] ──→ Linear(2048→256) + ReLU
                                      ↓
[Clinical Features: 3-d] ──→ Linear(3→32) + ReLU
                                      ↓
                              Concatenate [288-d]
                                      ↓
                            Linear(288→128) + ReLU
                                      ↓
                            Linear(128→5) + Softmax
                                      ↓
                         [5 Class Probabilities]
```

**Architecture Details:**
- **Image Branch:** Linear(2048 → 256) + ReLU
- **Clinical Branch:** Linear(3 → 32) + ReLU  
- **Fusion Layer:** Linear(288 → 128) + ReLU + Dropout(0.5)
- **Output Layer:** Linear(128 → 5)

**Training Configuration:**
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  
- **Model Size:** 2.2 MB (lightweight and efficient)

---

### 5. Risk Mapping & Clinical Recommendations
Based on predicted DR severity, the system provides evidence-based follow-up recommendations:

| DR Grade         | Risk Level | Follow-Up Recommendation                                      |
|------------------|------------|---------------------------------------------------------------|
| No DR            | Low        | Annual screening recommended. Maintain good glycemic control. |
| Mild NPDR        | Low        | Follow-up in 9-12 months. Monitor blood sugar and BP closely.|
| Moderate NPDR    | Moderate   | Follow-up in 6-9 months. Consider referral to ophthalmologist.|
| Severe NPDR      | High       | Follow-up in 3-4 months. Urgent ophthalmologist referral.    |
| Proliferative DR | Critical   | Immediate ophthalmologist referral required. High risk of vision loss.|

This ensures **clinical interpretability** and **real-world utility** of predictions.

---

### 6. Gradio Web Interface (Deployed on Hugging Face)
An interactive **Gradio** application enables real-time testing and validation:

**Features:**
- 📸 Upload retinal fundus images (JPG, PNG)
- 📋 Input clinical parameters via intuitive sliders
- 🔍 One-click analysis with instant results
- 📊 Probability distribution visualization
- 💡 Automated follow-up recommendations
- 📱 Mobile-friendly responsive design

**Deployment:**
- Hosted on **Hugging Face Spaces** (free tier)
- Powered by **Gradio 4.16.0**
- CPU-based inference (no GPU required)
- Public access with shareable link

---

## 🧠 How the Fusion DR Model Works

### 🔹 Step 1: Input Collection
The user provides two types of input:
- 🖼️ **Retinal Fundus Image** (color photograph of the retina)
- 🧪 **Clinical Data**:
  - HbA1c level (4.0-15.0%)
  - Systolic Blood Pressure (80-200 mmHg)
  - Duration of Diabetes (0-50 years)

These inputs represent both visual and physiological indicators of diabetic retinopathy.

---

### 🔹 Step 2: Image Preprocessing & Feature Extraction
- The uploaded image is resized to **224×224 pixels**
- Normalized using **ImageNet statistics** (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Passed through **pretrained ResNet50** (frozen weights)
- Outputs a **2048-dimensional feature vector** capturing deep visual patterns

> This step uses **transfer learning** – leveraging a model trained on millions of images to extract meaningful features from medical images.

---

### 🔹 Step 3: Clinical Data Formatting
- The three clinical values are converted into a **tensor** of shape `[1, 3]`
- No normalization applied (used as-is)
- These features represent the patient's metabolic and cardiovascular health

---

### 🔹 Step 4: Feature Fusion
- **2048-d image features** → processed by Image Branch → **256-d**
- **3-d clinical features** → processed by Clinical Branch → **32-d**
- Concatenated into a **288-dimensional fused vector**
- This represents a holistic view: what the eye shows + what the body reports

---

### 🔹 Step 5: Prediction via Fusion Model
- The 288-d fused vector passes through fully connected layers
- Outputs a **5-class probability vector** via softmax
- Each value represents the likelihood of one DR grade

Example output:
```
No DR: 5%  |  Mild: 10%  |  Moderate: 70%  |  Severe: 10%  |  PDR: 5%
→ Prediction: Moderate NPDR (70% confidence)
```

---

### 🔹 Step 6: Risk Assessment & Recommendations
- The predicted class is mapped to a risk level
- Clinical follow-up recommendations are generated
- Results are displayed with confidence scores

---

### 🔹 Step 7: Output Display
The app displays:
- ✅ **Predicted DR Severity** (e.g., Moderate NPDR)
- 📊 **Confidence Score** (e.g., 70.0%)
- 📈 **Probability Distribution** (bar chart of all 5 classes)
- 📋 **Follow-Up Recommendation** (e.g., "Follow-up in 6-9 months")
- 🔬 **Clinical Input Summary** (HbA1c, BP, Duration)

---

## 📦 Repository Structure

```
fusion-dr-predicator/
├── app.py                    # Gradio web interface (deployed)
├── fusion_model.py           # Fusion model architecture definition
├── fusion_model_mvp.pth      # Trained model weights (2.2 MB)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── dr-predict(1).ipynb       # Training notebook (Kaggle)
```

---

## 🚀 Deployment

### Dual Deployment Strategy

#### 1. **Hugging Face Spaces (Model Demo)**
- **Platform:** Hugging Face Spaces
- **URL:** [https://huggingface.co/spaces/MudLegacy/retinacare-dr-classifier](https://huggingface.co/spaces/MudLegacy/retinacare-dr-classifier)
- **Framework:** Gradio 4.16.0
- **Purpose:** Public model demonstration and testing
- **Compute:** CPU (free tier)
- **Accessibility:** Open to all users for evaluation

#### 2. **Full-Stack Web Application (Production)**
- **Status:** Under development by RetinaCare team
- **Architecture:** ML model deployed as backend API service
- **Purpose:** Production-ready healthcare application
- **Features:** 
  - Complete patient management system
  - Medical records integration
  - Healthcare provider dashboard
  - Secure authentication and data handling
  - HIPAA-compliant infrastructure (planned)

### Local Setup

1. **Clone the repository:**
```bash
git clone https://github.com/RetinaCare/classification-model.git
cd classification-model/fusion-dr-predicator
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
python app.py
```

The app will launch at `http://localhost:7860`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch 2.1.0, Torchvision 0.16.0 |
| **Pretrained Model** | ResNet50 (ImageNet weights) |
| **Web Interface** | Gradio 4.16.0 |
| **Deployment** | Hugging Face Spaces |
| **Image Processing** | Pillow 10.2.0 |
| **Numerical Computing** | NumPy 1.24.3 |

---

## 📊 Model Performance

- **Architecture:** ResNet50 (feature extractor) + Custom Fusion Layers
- **Model Size:** 2.2 MB (highly efficient)
- **Input:** 224×224 RGB images + 3 clinical features
- **Output:** 5-class probability distribution
- **Inference Speed:** Real-time (<2 seconds per prediction)

---

## 💻 Usage Guide

### Using the Web Interface

#### Option 1: Hugging Face Demo
1. **Visit:** [https://huggingface.co/spaces/MudLegacy/retinacare-dr-classifier](https://huggingface.co/spaces/MudLegacy/retinacare-dr-classifier)
2. **Upload Image:** Click to upload a retinal fundus photograph
3. **Input Clinical Data:**
   - Adjust HbA1c slider (4.0-15.0%)
   - Set Blood Pressure (80-200 mmHg)
   - Enter Diabetes Duration (0-50 years)
4. **Analyze:** Click "🔍 Analyze Retinal Image"
5. **Review Results:** View prediction, confidence, and recommendations

#### Option 2: Full Web Application (Coming Soon)
Access the complete healthcare platform with patient management, medical records, and integrated diagnostics.

---

## ⚠️ Disclaimer

This is an **AI-assisted diagnostic tool for research and educational purposes**. It should **not replace professional medical judgment**. Always consult with a qualified ophthalmologist for clinical decisions.

The model is designed to support healthcare providers in screening and risk assessment, not to provide definitive diagnoses.

---

## 👥 Contributors

### Machine Learning & Model Development
- **Almustapha Damilola Usman** – *ML Engineer*  
  - Model architecture design and implementation
  - Dataset preparation and preprocessing
  - Model training and optimization
  - Hugging Face deployment
  - Model evaluation and validation

### Web Application Development Team
- **RetinaCare Development Team**
  - Full-stack web application development
  - Backend API integration
  - Frontend user interface
  - Database architecture
  - Security and compliance implementation
  - Production deployment and scaling

---

## 🔮 Future Improvements

### Model Enhancements
- [ ] **Explainable AI:** Integrate Grad-CAM visualizations to highlight DR-related regions
- [ ] **Model Enhancement:** Fine-tune on larger, more diverse datasets
- [ ] **Ensemble Methods:** Combine multiple models for improved accuracy
- [ ] **Extended Clinical Data:** Incorporate additional patient history and biomarkers

### Application Features
- [ ] **Multi-language Support:** Expand interface to support multiple languages
- [ ] **Mobile App:** Develop native iOS/Android applications
- [ ] **Real-time Monitoring:** Enable longitudinal tracking of patient DR progression
- [ ] **Batch Processing:** Support bulk image analysis for screening programs
- [ ] **Integration APIs:** REST API and webhooks for EHR/EMR system integration

### Production & Scaling
- [ ] **HIPAA Compliance:** Full medical data security and privacy compliance
- [ ] **Cloud Infrastructure:** Scalable deployment on AWS/GCP/Azure
- [ ] **Performance Optimization:** GPU acceleration and model quantization
- [ ] **Clinical Validation:** Multi-center clinical trials and FDA approval pathway

---

## 📄 License

[Add your license here - e.g., MIT, Apache 2.0]

---

## 📞 Contact & Support

For questions, feedback, or collaboration:
- **Model Demo:** [Hugging Face Space](https://huggingface.co/spaces/MudLegacy/retinacare-dr-classifier)
- **GitHub Repository:** [RetinaCare Classification Model](https://github.com/RetinaCare/classification-model)
- **ML Engineer:** Almustapha Damilola Usman
- **Project Lead:** [RetinaCare Team Contact]

---

## 🙏 Acknowledgments

- **Dataset:** EyePACS, APTOS & Messidor teams for providing comprehensive DR datasets
- **PyTorch & Torchvision:** For robust deep learning framework and pretrained models
- **Hugging Face:** For free hosting and excellent deployment infrastructure
- **Gradio:** For intuitive interface development tools

---

> *RetinaCare aims to bridge the gap between medical imaging and intelligent decision support – ensuring timely and accurate DR diagnosis for better patient outcomes.*

**Built with ❤️ for accessible healthcare**