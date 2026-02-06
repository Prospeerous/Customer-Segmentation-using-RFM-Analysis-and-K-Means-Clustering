# Customer Segmentation using RFM Analysis & K-Means Clustering

An end-to-end data science project that segments customers based on their purchasing behaviour using **RFM (Recency, Frequency, Monetary)** analysis and **K-Means clustering**, with a Flask web application for real-time segment prediction.

## Project Overview

This project analyses transactional data from the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset (UCI Machine Learning Repository) to identify distinct customer segments and deliver actionable business recommendations for targeted marketing strategies.

### Key Objectives
- Compute RFM metrics per customer from raw transaction data
- Determine optimal clusters using the Elbow Method and Silhouette Score analysis
- Profile each segment with business-relevant labels and strategic recommendations
- Deploy a lightweight Flask app for real-time customer classification

## Methodology

| Stage | Description |
|-------|-------------|
| Data Cleaning | Removed nulls, cancelled orders, and invalid transactions |
| Feature Engineering | Computed Recency, Frequency, Monetary values and Customer Lifetime Value (CLV) |
| Scaling | StandardScaler normalisation for fair distance-based clustering |
| Clustering | K-Means (k=4, optimal via silhouette analysis) and DBSCAN comparison |
| Visualisation | PCA-projected 2D cluster plots, segment distributions, CLV analysis |
| Deployment | Flask web app with pre-trained model for real-time prediction |

## Customer Segments

| Segment | Profile |
|---------|---------|
| **Champions** | High-value, frequent, recent buyers — reward with VIP programs |
| **Loyal Customers** | Consistent purchasers — upsell and cross-sell opportunities |
| **New / Potential** | Recent first-time buyers — nurture with onboarding campaigns |
| **At Risk / Hibernating** | Lapsed customers — re-engage with win-back offers |

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn, matplotlib, seaborn
- **Flask** — web application framework
- **scikit-learn** — K-Means, DBSCAN, PCA, StandardScaler
- **joblib** — model serialisation

## Project Structure

```
├── customer-segmentation-using-rfm.ipynb   # Full analysis notebook
├── app.py                                  # Flask web application
├── kmeans_model.pkl                        # Trained K-Means model
├── scaler.pkl                              # Fitted StandardScaler
├── templates/
│   └── index.html                          # Web UI template
├── requirements.txt                        # Python dependencies
└── README.md
```

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Prospeerous/Customer-Segmentation-using-RFM-Analysis-and-K-Means-Clustering.git
cd Customer-Segmentation-using-RFM-Analysis-and-K-Means-Clustering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Results Summary

- **4,312 customers** segmented from 407,664 cleaned transactions
- **Optimal k=4** clusters validated with silhouette score of **0.610**
- Segments mapped to actionable business strategies for retention, growth, and re-engagement

## License

This project is open source and available under the [MIT License](LICENSE).
