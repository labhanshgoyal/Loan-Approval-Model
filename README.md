┌─────────────────────────────────────────────────────────────────┐
│                        RAW DATA INGESTION                        │
│                     (df1_loan.xlsx loaded)                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EXPLORATORY DATA ANALYSIS (EDA)                 │
│   • Distribution plots       • Correlation heatmap              │
│   • Class balance check       • Outlier detection               │
│   • Feature-target relationships                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DATA PREPROCESSING                         │
│   • Handle missing values (imputation)                          │
│   • Encode categorical features (Label / One-Hot Encoding)      │
│   • Feature scaling (StandardScaler / MinMaxScaler)             │
│   • Train/Test split (80% / 20%)                                │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MODEL TRAINING                            │
│                                                                  │
│   ┌──────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│   │ Logistic         │  │ Random          │  │   XGBoost    │  │
│   │ Regression       │  │ Forest          │  │  Classifier  │  │
│   └──────────────────┘  └─────────────────┘  └──────────────┘  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL EVALUATION                           │
│   • Accuracy Score           • Confusion Matrix                 │
│   • Precision / Recall       • F1-Score                         │
│   • ROC-AUC Curve            • Feature Importance Plot          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INSIGHTS & CONCLUSIONS                        │
│   • Best model selection     • Key feature drivers              │
│   • Business recommendations • Deployment considerations         │
└─────────────────────────────────────────────────────────────────┘
