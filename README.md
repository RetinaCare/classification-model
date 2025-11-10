Fusion DR Predictor (MVP)
Project Overview
This project is a machine learning-based diagnostic tool designed to predict the severity of Diabetic Retinopathy (DR) by fusing image features from retinal scans with clinical data. It supports early detection and personalized follow-up recommendations.

Model Development Workflow
1. Problem Definition
We aimed to build a model that classifies DR severity into five categories:
- No DR
- Mild NPDR
- Moderate NPDR
- Severe NPDR
- Proliferative DR (PDR)
The model also provides a follow-up recommendation based on the predicted risk level.

2. Data Sources
- Image Data: Retinal fundus images (preprocessed and resized)
- Clinical Data: Three key features:
- HbA1c level
- Blood pressure
- Duration of diabetes

3. Image Feature Extraction
We used a pretrained ResNet50 model from torchvision to extract deep features from retinal images:
- Removed the final classification layer (resnet.fc = Identity())
- Output: a 2048-dimensional feature vector per image
This approach leverages transfer learning to capture rich visual patterns without retraining the entire CNN.

4. Fusion Model Architecture
The fusion model combines image and clinical features into a single prediction pipeline:
[Image Features: 2048-d] + [Clinical Features: 3-d]
            ↓
     Concatenated Vector
            ↓
     Fully Connected Layers
            ↓
     Softmax Output (5 classes)


The model is trained using cross-entropy loss and optimized with Adam.

5. Risk Mapping Logic
We created a separate module (risk_mapper.py) that maps each predicted class to a follow-up recommendation:
| DR GRADE | Risk level | Follow up Recommendation | 
| No DR | Low | 12 Months | 
| Mild NPDR | Low | 12 Months | 
| Moderate NPDR | Moderate | 6 Months | 
| Severe NPDR | High | 3 Months | 
| PDR | Critical | Immediate Referral | 


This logic ensures clinical relevance and supports decision-making.

6. Streamlit MVP Interface
We built a simple Streamlit app (app.py) that allows users to:
- Upload a retinal image
- Input clinical values
- Run prediction
- View DR grade, probability vector, and follow-up recommendation
This interface is ideal for testing and showcasing the model.

📦 Project Structure
fusion-dr-predictor/
├── app.py                  # Streamlit interface ( Will be updated )
├── fusion_model.py         # Fusion model definition
├── utils.py                # ResNet feature extraction
├── risk_mapper.py          # Risk logic
├── fusion_model_mvp.pth    # Trained model weights
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation



👥 Contributors
- ALMUSTAPHA DAMILOLA USMAN — Machine Learning Engineer (Model design, training, integration)
- Team Members — Web development, UI/UX, deployment (to be added post-launch)

