# 🧠 Fusion DR Predictor (MVP)

## 📖 Project Overview
**Fusion DR Predictor** is a machine learning-based diagnostic tool designed to **predict the severity of Diabetic Retinopathy (DR)** by fusing retinal image features with patient clinical data.  
This MVP (Minimum Viable Product) supports **early detection**, **risk assessment**, and **personalized follow-up recommendations**, empowering healthcare providers to make informed decisions.

---

## 🏗️ Model Development Workflow

### 1. Problem Definition
The model classifies Diabetic Retinopathy (DR) severity into five categories:
- **No DR**
- **Mild NPDR**
- **Moderate NPDR**
- **Severe NPDR**
- **Proliferative DR (PDR)**

Additionally, the system provides **follow-up recommendations** based on predicted risk levels.

---

### 2. Data Sources
- **Image Data:** Retinal fundus images (preprocessed and resized)  
- **Clinical Data:** Three key patient features:
  - HbA1c level  
  - Blood pressure  
  - Duration of diabetes  

📂 **Dataset Used:** [EyePACS, APTOS & Messidor Diabetic Retinopathy Dataset](https://www.kaggle.com/datasets/ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy)

This combined dataset provides a diverse and comprehensive collection of retinal images suitable for building and validating DR severity models.

---

### 3. Image Feature Extraction
A **pretrained ResNet50** model from `torchvision` was utilized to extract deep image features:
- The final classification layer (`resnet.fc`) was replaced with `Identity()`
- The resulting feature vector per image is **2048-dimensional**

This approach leverages **transfer learning**, allowing the model to capture rich visual representations without retraining the full CNN.

---

### 4. Fusion Model Architecture
The model fuses **image** and **clinical** features into a single prediction pipeline:

```
[Image Features: 2048-d] + [Clinical Features: 3-d]
              ↓
       Concatenated Vector
              ↓
   Fully Connected Layers
              ↓
        Softmax Output (5 Classes)
```

The model is trained using:
- **Loss Function:** Cross-Entropy Loss  
- **Optimizer:** Adam  

---

### 5. Risk Mapping Logic
A custom module, `risk_mapper.py`, maps predicted classes to actionable follow-up recommendations:

| DR Grade         | Risk Level | Follow-Up Recommendation   |
|------------------|-------------|-----------------------------|
| No DR            | Low         | 12 Months                  |
| Mild NPDR        | Low         | 12 Months                  |
| Moderate NPDR    | Moderate    | 6 Months                   |
| Severe NPDR      | High        | 3 Months                   |
| PDR              | Critical    | Immediate Referral          |

This ensures **clinical interpretability** and **real-world utility** of predictions.

---

### 6. Streamlit MVP Interface
A simple **Streamlit** application (`app.py`) enables interactive testing:
- Upload a retinal image  
- Input clinical parameters  
- Run prediction  
- View:
  - DR Grade  
  - Probability Vector  
  - Follow-Up Recommendation  

This interface is ideal for **demo**, **validation**, and **stakeholder feedback**.

---

## 🧠 Step-by-Step: How the Fusion DR Model Works

### 🔹 Step 1: Input Collection
The user provides two types of input:
- 🖼️ **Retinal Image** (uploaded via the app)
- 🧪 **Clinical Data**:
  - HbA1c level
  - Blood pressure
  - Duration of diabetes

These inputs represent both visual and physiological indicators of diabetic retinopathy.

---

### 🔹 Step 2: Image Feature Extraction (ResNet50)
- The uploaded image is resized and normalized.
- It is passed through a **pretrained ResNet50** model (with the final classification layer removed).
- The model outputs a **2048-dimensional feature vector** that captures deep visual patterns in the retina (e.g., microaneurysms, hemorrhages).

> This step uses **transfer learning** — leveraging a model trained on millions of images to extract meaningful features from medical images.

---

### 🔹 Step 3: Clinical Data Formatting
- The three clinical values are converted into a **tensor** of shape `[1, 3]`.
- These features are numerical and represent the patient’s metabolic and cardiovascular health.

---

### 🔹 Step 4: Feature Fusion
- The **2048-d image vector** and the **3-d clinical vector** are **concatenated** into a single `[1, 2051]` vector.
- This fused vector represents a holistic view of the patient: both what the eye shows and what the body reports.

---

### 🔹 Step 5: Prediction via Fusion Model
- The fused vector is passed through a **fully connected neural network** (custom PyTorch model).
- The model outputs a **5-class probability vector** using a **softmax layer**.
- Each value represents the likelihood of one of the DR grades.

Example output:
```
[0.05, 0.10, 0.70, 0.10, 0.05]
→ Highest probability: Moderate NPDR
```

---

### 🔹 Step 6: Risk Mapping
- The predicted class index is passed to the `risk_mapper.py` module.
- This maps the class to:
  - A **risk level** (Low, Moderate, High, Critical)
  - A **follow-up recommendation** (e.g., 12 months, 6 months, immediate referral)

---

### 🔹 Step 7: Output Display
The app displays:
- ✅ Predicted DR Grade (e.g., Moderate NPDR)
- 📊 Probability Vector (e.g., `[0.05, 0.10, 0.70, 0.10, 0.05]`)
- 📅 Follow-Up Recommendation (e.g., “6 Months”)

This output is designed to be **clinically interpretable** and **actionable**.

---

## 📦 Project Structure

```
fusion-dr-predictor/
├── app.py                  # Streamlit interface
├── fusion_model.py         # Fusion model definition
├── utils.py                # ResNet feature extraction utilities
├── risk_mapper.py          # Risk mapping logic
├── fusion_model_mvp.pth    # Trained model weights
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 👥 Contributors
- **Almustapha Damilola Usman** — *Machine Learning Engineer*  
  *(Model design, training, and integration)*  
- **Team Members** — *Web Development, UI/UX, and Deployment*  
  *(To be added post-launch)*

---

## 🚀 Future Improvements
- Integration with cloud-based medical imaging APIs  
- Deployment to scalable web infrastructure  
- Incorporation of explainable AI (Grad-CAM visualizations)  
- Model fine-tuning with larger, diverse datasets  

---

## 🧩 Tech Stack
- **Language:** Python  
- **Libraries:** PyTorch, Torchvision, Streamlit, NumPy, Pandas  
- **Model:** ResNet50 (Transfer Learning)  
- **Interface:** Streamlit  
- **Deployment Target:** MVP (Web Demo)

---

> *Fusion DR Predictor aims to bridge the gap between medical imaging and intelligent decision support — ensuring timely and accurate DR diagnosis for better patient outcomes.*
