from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Map clusters → business segments with insights and recommendations
segment_data = {
    0: {
        'name': 'Champions',
        'icon': 'trophy',
        'color': '#28a745',
        'description': 'Your most valuable customers who bought recently, buy often, and spend the most.',
        'insights': [
            'Highest lifetime value among all segments',
            'Strong brand loyalty and frequent repeat purchases',
            'Most likely to promote your brand through word-of-mouth',
        ],
        'recommendations': [
            'Reward them with exclusive loyalty programs and VIP perks',
            'Offer early access to new products and special editions',
            'Invite them to provide reviews and referrals',
            'Avoid over-discounting — they already buy at full price',
        ],
    },
    1: {
        'name': 'Loyal Customers',
        'icon': 'heart',
        'color': '#0d6efd',
        'description': 'Consistent buyers who purchase regularly and contribute reliable revenue.',
        'insights': [
            'Steady purchase frequency with good monetary contribution',
            'Reliable revenue stream for the business',
            'High potential to become Champions with the right engagement',
        ],
        'recommendations': [
            'Upsell and cross-sell higher-value or complementary products',
            'Launch a referral program to leverage their satisfaction',
            'Offer membership or subscription-based benefits',
            'Gather feedback to improve their experience further',
        ],
    },
    2: {
        'name': 'At Risk / Hibernating',
        'icon': 'exclamation-triangle',
        'color': '#dc3545',
        'description': 'Previously active customers who haven\'t purchased recently and may be churning.',
        'insights': [
            'Declining engagement — last purchase was a long time ago',
            'Were once valuable but are now drifting away',
            'High risk of permanent churn if not re-engaged promptly',
        ],
        'recommendations': [
            'Launch targeted win-back campaigns with personalised offers',
            'Send re-engagement emails highlighting new products or improvements',
            'Offer time-limited discounts to create urgency',
            'Survey them to understand why they stopped purchasing',
        ],
    },
    3: {
        'name': 'New / Potential Customers',
        'icon': 'seedling',
        'color': '#fd7e14',
        'description': 'Recent customers with low frequency — they are still exploring your brand.',
        'insights': [
            'Made a recent purchase but haven\'t established buying habits yet',
            'High growth potential if nurtured correctly',
            'Sensitive to first impressions and onboarding experience',
        ],
        'recommendations': [
            'Create a strong onboarding experience with welcome offers',
            'Send educational content about your products and brand story',
            'Encourage a second purchase with a follow-up discount',
            'Provide excellent customer support to build early trust',
        ],
    },
}

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])

        X = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(X)[0]

        result = segment_data[cluster]
        result['recency'] = recency
        result['frequency'] = frequency
        result['monetary'] = monetary

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
