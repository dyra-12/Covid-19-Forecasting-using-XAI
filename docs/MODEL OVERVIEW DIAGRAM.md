                        ┌────────────────────────────┐
                        │     Raw COVID-19 Data       │
                        │  (cases, dates, features)   │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌────────────────────────────┐
                        │      Data Preprocessing     │
                        │  • Cleaning & scaling       │
                        │  • Supervised windowing     │
                        │  • Train/validation split   │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                   ┌──────────────────────────────────────────┐
                   │          Deep Learning Models             │
                   │──────────────────────────────────────────│
                   │  LSTM            → seq dependencies       │
                   │  BiLSTM          → bidirectional memory   │
                   │  RNN             → basic recurrent model  │
                   │  CNN             → local temporal patterns │
                   │  MLP             → baseline dense model    │
                   │  Hybrid Models:                            │
                   │     • CNN → BiLSTM                         │
                   │     • CNN → LSTM                           │
                   │     • MLP → CNN → LSTM                     │
                   │     • CNN → Dense fusion                   │
                   └──────────────┬─────────────────────────────┘
                                   │
                                   ▼
                        ┌────────────────────────────┐
                        │        Model Output         │
                        │   Predicted case numbers    │
                        │   (forecasted time steps)   │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                ┌───────────────────────────────────────────────┐
                │         Explainability (XAI Layer)             │
                │───────────────────────────────────────────────│
                │  SHAP:                                        │
                │     • Global feature importance               │
                │     • Local explanations                      │
                │     • Contribution of each input window       │
                │                                               │
                │  LIME:                                        │
                │     • Instance-level interpretability         │
                │     • Highlighting time-step influences       │
                └──────────────┬────────────────────────────────┘
                               │
                               ▼
                    ┌────────────────────────────┐
                    │     Evaluation Outputs      │
                    │  • RMSE, MAE, MAPE          │
                    │  • Actual vs Predicted plot │
                    │  • SHAP summary plot        │
                    │  • LIME explanation charts  │
                    └────────────────────────────┘
