import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data', 'bengaluru_retail_transactions.csv')
MODEL = os.path.join(BASE, 'models', 'store_location_model.pkl')

rng = np.random.default_rng(42)
areas = {
    'Electronic City': 1.30, 'Koramangala': 1.38, 'Whitefield': 1.34,
    'Indiranagar': 1.35, 'HSR Layout': 1.28, 'Marathahalli': 1.24,
    'BTM Layout': 1.18, 'Jayanagar': 1.08, 'JP Nagar': 1.05,
    'Yelahanka': 0.96, 'Hebbal': 1.00, 'Malleshwaram': 1.02,
    'Rajajinagar': 1.04, 'Banashankari': 0.98, 'KR Puram': 1.12,
    'Bellandur': 1.32, 'Sarjapur Road': 1.27, 'HSR Sector 1': 1.20,
    'MG Road': 1.42, 'BTM 2nd Stage': 1.14
}
categories = {
    'Electronics': 1.12, 'Grocery': 1.20, 'Fashion': 1.05,
    'Beauty & Personal Care': 0.96, 'Pharmacy': 1.10,
    'Home & Kitchen': 0.93, 'Sports & Fitness': 0.88,
    'Food & Beverages': 1.25, 'Books & Stationery': 0.82
}
rows = []
months = pd.date_range('2025-01-01', '2026-09-01', freq='MS')
for area, area_factor in areas.items():
    for category, cat_factor in categories.items():
        for month in months:
            base_orders = 105 * area_factor * cat_factor
            season = 1 + 0.10*np.sin(month.month/12*2*np.pi) + rng.normal(0, 0.05)
            orders = max(8, int(base_orders * season + rng.normal(0, 12)))
            customers = max(5, int(orders * rng.uniform(0.48, 0.78)))
            avg_spend = max(80, rng.normal({'Electronics':1800,'Grocery':650,'Fashion':1200,'Beauty & Personal Care':800,'Pharmacy':550,'Home & Kitchen':1000,'Sports & Fitness':900,'Food & Beverages':420,'Books & Stationery':450}[category], 90))
            avg_qty = max(1, rng.normal(2.1 if category in ['Grocery','Food & Beverages'] else 1.45, 0.35))
            repeat_rate = np.clip(rng.normal(0.34 + 0.06*area_factor, 0.07), 0.08, 0.80)
            digital_ratio = np.clip(rng.normal(0.72, 0.10), 0.25, 0.98)
            revenue = orders * avg_spend
            rows.append([month.date(), area, category, orders, customers, round(avg_spend,2), round(avg_qty,2), round(repeat_rate,3), round(digital_ratio,3), round(revenue,2)])

df = pd.DataFrame(rows, columns=['month','area','category','orders','customers','avg_spend','avg_quantity','repeat_rate','digital_payment_ratio','revenue'])
df.to_csv(DATA, index=False)

# Build training rows from each area/category/month. Target is whether the next-month demand is strong.
df['next_orders'] = df.groupby(['area','category'])['orders'].shift(-1)
train = df.dropna().copy()
train['demand_growth'] = (train['next_orders'] - train['orders']) / train['orders']
train['target'] = ((train['next_orders'] >= train['orders']*0.95) & (train['repeat_rate'] >= 0.30) & (train['customers'] >= 55)).astype(int)
features = ['orders','customers','avg_spend','avg_quantity','repeat_rate','digital_payment_ratio','revenue']
X = train[features]
y = train['target']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=250, max_depth=10, random_state=42, class_weight='balanced')
model.fit(Xtr, ytr)
pred = model.predict(Xte)
print('Accuracy:', round(accuracy_score(yte, pred), 4))
print(classification_report(yte, pred))
joblib.dump({'model': model, 'features': features, 'areas': list(areas.keys()), 'categories': list(categories.keys())}, MODEL)
print('Saved:', MODEL)
