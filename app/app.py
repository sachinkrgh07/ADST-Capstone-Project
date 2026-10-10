import os, sqlite3, joblib, numpy as np, pandas as pd
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB=os.path.join(BASE,'data','smartstore.db')
DATA_PATH=os.path.join(BASE,'data','bengaluru_retail_transactions.csv')
MODEL_PATH=os.path.join(BASE,'models','store_location_model.pkl')
os.makedirs(os.path.dirname(DB),exist_ok=True)
app=Flask(__name__,template_folder='../templates',static_folder='../static')
app.secret_key='smartstore-ai-local-2026'
AREAS=['Electronic City','Koramangala','Whitefield','Indiranagar','HSR Layout','Marathahalli','BTM Layout','Jayanagar','JP Nagar','Yelahanka','Hebbal','Malleshwaram','Rajajinagar','Banashankari','KR Puram','Bellandur','Sarjapur Road','HSR Sector 1','MG Road','BTM 2nd Stage']
CATEGORIES=['Electronics','Grocery','Fashion','Beauty & Personal Care','Pharmacy','Home & Kitchen','Sports & Fitness','Food & Beverages','Books & Stationery']

def conn():
    c=sqlite3.connect(DB); c.row_factory=sqlite3.Row; return c

def init_db():
    c=conn()
    c.execute('''CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,name TEXT NOT NULL,email TEXT UNIQUE NOT NULL,password TEXT NOT NULL,role TEXT DEFAULT 'user',created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    # Upgrade older database created by the previous version.
    cols=[r['name'] for r in c.execute('PRAGMA table_info(users)').fetchall()]
    if 'role' not in cols: c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
    c.execute('''CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER,category TEXT,area TEXT,recommendation TEXT,probability REAL,demand_score REAL,monthly_orders INTEGER,monthly_revenue REAL,monthly_profit REAL,created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''INSERT OR IGNORE INTO users(name,email,password,role) VALUES(?,?,?,?)''',('Administrator','admin@smartstore.ai',generate_password_hash('admin123'),'admin'))
    c.commit(); c.close()
init_db()
BUNDLE=joblib.load(MODEL_PATH); MODEL=BUNDLE['model']; FEATURES=BUNDLE['features']; DF=pd.read_csv(DATA_PATH)

def auth(f):
    @wraps(f)
    def w(*a,**kw):
        if 'user_id' not in session: return redirect(url_for('login',next=request.path))
        return f(*a,**kw)
    return w

def admin_only(f):
    @wraps(f)
    def w(*a,**kw):
        if session.get('role')!='admin': flash('Admin access required.','error'); return redirect(url_for('dashboard'))
        return f(*a,**kw)
    return w

def metrics(area,cat):
    d=DF[(DF.area==area)&(DF.category==cat)]
    if d.empty: return None
    return {'orders':float(d.orders.mean()),'customers':float(d.customers.mean()),'avg_spend':float(d.avg_spend.mean()),'avg_quantity':float(d.avg_quantity.mean()),'repeat_rate':float(d.repeat_rate.mean()),'digital_payment_ratio':float(d.digital_payment_ratio.mean()),'revenue':float(d.revenue.mean())}

def predict_area(area,cat,rent=65000,margin=.22):
    m=metrics(area,cat)
    if not m:return None
    x=pd.DataFrame([[m[f] for f in FEATURES]],columns=FEATURES)
    prob=float(MODEL.predict_proba(x)[0][1])*100

    # Rank the selected category across Bengaluru areas.
    # IMPORTANT: aggregate all monthly records into one row per area.
    cat_rows=DF[DF["category"].astype(str).str.strip().str.lower()==cat.lower()].copy()

    if cat_rows.empty:
        category_demand=0.0
    else:
        numeric_cols=["orders","customers","avg_spend","repeat_rate","digital_payment_ratio"]
        for col in numeric_cols:
            cat_rows[col]=pd.to_numeric(cat_rows[col],errors="coerce").fillna(0)

        if cat_rows["repeat_rate"].max()>1:
            cat_rows["repeat_rate"]=cat_rows["repeat_rate"]/100
        if cat_rows["digital_payment_ratio"].max()>1:
            cat_rows["digital_payment_ratio"]=cat_rows["digital_payment_ratio"]/100

        cat_rows=cat_rows.groupby("area",as_index=False)[
            ["orders","customers","avg_spend","repeat_rate","digital_payment_ratio"]
        ].mean()

        selected_row=cat_rows[
            cat_rows["area"].astype(str).str.strip().str.lower()==area.lower()
        ]

        if selected_row.empty:
            category_demand=0.0
        else:
            selected_row=selected_row.iloc[0]

            def pct(col):
                values=cat_rows[col]
                if len(values)<=1:
                    return 0.5
                return float((values<=selected_row[col]).mean())

            category_demand=(
                pct("orders")*0.38+
                pct("customers")*0.22+
                pct("avg_spend")*0.14+
                pct("repeat_rate")*0.16+
                pct("digital_payment_ratio")*0.10
            )*100

    score=float(np.clip(prob*.40+category_demand*.60,0,100))
    monthly_orders=int(m['orders']*(1.02+category_demand/450))
    revenue=monthly_orders*m['avg_spend']
    profit=revenue*margin-rent
    rec='OPEN STORE' if score>=72 else ('CONSIDER' if score>=52 else 'AVOID FOR NOW')
    status='Strong opportunity' if score>=72 else ('Moderate opportunity' if score>=52 else 'Low opportunity')
    return {**m,'area':area,'category':cat,'probability':round(prob,1),
            'demand_score':round(score,1),'monthly_orders':monthly_orders,
            'monthly_revenue':revenue,'monthly_profit':profit,
            'recommendation':rec,'status':status}

@app.context_processor
def ctx(): return {'current_user':session.get('name'),'current_role':session.get('role'),'areas':AREAS,'categories':CATEGORIES}

@app.route('/')
def index():
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method=='POST':
        name=request.form['name'].strip(); email=request.form['email'].strip().lower(); password=request.form['password']
        if len(password)<6: flash('Password must be at least 6 characters.','error'); return render_template('signup.html')
        c=conn()
        try:
            c.execute('INSERT INTO users(name,email,password,role) VALUES(?,?,?,?)',(name,email,generate_password_hash(password),'user')); c.commit()
            flash('Account created. Please sign in with your email and password.','success'); return redirect(url_for('login'))
        except sqlite3.IntegrityError: flash('This email is already registered.','error')
        finally:c.close()
    return render_template('signup.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        email=request.form['email'].strip().lower(); password=request.form['password']; c=conn(); u=c.execute('SELECT * FROM users WHERE email=?',(email,)).fetchone(); c.close()
        if u and check_password_hash(u['password'],password):
            session.clear(); session.update(user_id=u['id'],name=u['name'],role=u['role'],email=u['email'])
            return redirect(request.args.get('next') or url_for('dashboard'))
        flash('Incorrect email or password.','error')
    return render_template('login.html')

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('index'))

@app.route('/dashboard')
@auth
def dashboard():
    c=conn(); recent=c.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC LIMIT 5',(session['user_id'],)).fetchall(); count=c.execute('SELECT COUNT(*) n FROM predictions WHERE user_id=?',(session['user_id'],)).fetchone()['n']; c.close()
    return render_template('dashboard.html',recent=recent,prediction_count=count)

@app.route('/predict',methods=['GET','POST'])
@auth
def predict():
    results=None; selected=None; rent=65000; margin=22
    if request.method=='POST':
        selected=request.form['category']
        rent=float(request.form.get('rent',65000))
        margin=float(request.form.get('margin',22))
        results=[predict_area(a,selected,rent,margin/100) for a in AREAS]
        results=[x for x in results if x]
        results.sort(key=lambda x:(x['demand_score'],x['probability']),reverse=True)

        # Automatically save the best recommendation for this logged-in user.
        if results:
            top=results[0]
            c=conn()
            c.execute(
                "INSERT INTO predictions(user_id,category,area,recommendation,probability,demand_score,monthly_orders,monthly_revenue,monthly_profit) VALUES(?,?,?,?,?,?,?,?,?)",
                (session['user_id'],top['category'],top['area'],top['recommendation'],
                 top['probability'],top['demand_score'],top['monthly_orders'],
                 top['monthly_revenue'],top['monthly_profit'])
            )
            c.commit()
            c.close()
    return render_template('predict.html',results=results,selected=selected,rent=rent,margin=margin,categories=CATEGORIES)

@app.route('/save',methods=['POST'])
@auth
def save():
    d=request.get_json(); c=conn(); c.execute('''INSERT INTO predictions(user_id,category,area,recommendation,probability,demand_score,monthly_orders,monthly_revenue,monthly_profit) VALUES(?,?,?,?,?,?,?,?,?)''',(session['user_id'],d['category'],d['area'],d['recommendation'],d['probability'],d['demand_score'],d['monthly_orders'],d['monthly_revenue'],d['monthly_profit'])); c.commit(); c.close(); return jsonify(ok=True)

@app.route('/history')
@auth
def history():
    c=conn(); rows=c.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY id DESC',(session['user_id'],)).fetchall(); c.close(); return render_template('history.html',rows=rows)

@app.route('/analytics')
@auth
def analytics():
    cat=request.args.get('category',CATEGORIES[0]); ranked=sorted([predict_area(a,cat) for a in AREAS],key=lambda x:x['demand_score'],reverse=True)
    return render_template('analytics.html',ranked=ranked,selected=cat)

@app.route('/admin')
@admin_only
def admin():
    c=conn(); users=c.execute('SELECT id,name,email,role,created_at FROM users ORDER BY id DESC').fetchall(); preds=c.execute('''SELECT p.*,u.name user_name FROM predictions p LEFT JOIN users u ON u.id=p.user_id ORDER BY p.id DESC LIMIT 50''').fetchall(); total=c.execute('SELECT COUNT(*) n FROM predictions').fetchone()['n']; c.close()
    return render_template('admin.html',users=users,preds=preds,total=total)

@app.route('/about')
def about(): return render_template('about.html')

if __name__=='__main__': app.run(debug=True)
