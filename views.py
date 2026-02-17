from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pymysql
import os, joblib, base64, json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import shap
from lime.lime_tabular import LimeTabularExplainer
import seaborn as sns
from django.conf import settings
import traceback

# Create your views here.
def index(request):
    return render(request,'index.html')

def admin_login(request):
    return render(request,'admin/admin_login.html')

def admin_login_action(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    if username == 'Admin' and password == 'Admin':
        return render(request,'admin/admin_home.html')
    else:
        context={'data':'Login Failed..! Please Enter Valid Credentials..!'}
        return render(request,'admin/admin_login.html',context)


def logout(request):
    return render(request,'index.html')

def admin_home(request):
    return render(request,'admin/admin_home.html')


# upload dataset(data collection)
def upload_dataset(request):
    return render(request,'admin/upload_dataset.html')

global df
def upload_dataset_action(request):
    global df
    if request.method=='POST':
        filename=request.FILES['file']
        df=pd.read_csv(filename)
        dataset_len= len(df)
        columns = df.shape
        describe = df.describe()
        table_html=df.head(10).to_html(index=False, classes="custom-table")
        context={
            'msg':'Dataset Uploaded Successfully',
            'dataset_len':dataset_len,
            'columns' : columns,
            'describe':describe,
            'table_html':table_html
        }
        return render(request,'admin/upload_dataset.html',context)


def preprocess(request):

    raw_folder = "dataset/cyber_attack/"
    preprocessed_folder = "dataset/preprocessed_cyber_attack/"

    selected_columns = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Fwd Packet Length Mean",
        "Packet Length Std",
        "Min Packet Length",
        "Max Packet Length",
        "SYN Flag Count",
        "ACK Flag Count",
        "Label"
    ]

    selected_norm = [col.lower().strip() for col in selected_columns]

    # Create preprocessed folder if missing
    if not os.path.exists(preprocessed_folder):
        os.makedirs(preprocessed_folder)

    # Check if preprocessed files already exist
    processed_files = [f for f in os.listdir(preprocessed_folder) if f.endswith(".csv")]

    # ---------------------------------------------------------
    # 1️⃣ If preprocessed data already exists → load & skip work
    # ---------------------------------------------------------
    if len(processed_files) > 0:
        preprocessed_list = []

        for file in processed_files:
            df = pd.read_csv(os.path.join(preprocessed_folder, file))
            preprocessed_list.append(df)

        final_df = pd.concat(preprocessed_list, ignore_index=True)

        X = final_df.drop("label", axis=1)
        y = final_df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        context = {
            "msg": "⚠️ Data Preprocessed Successfully....!",
            "columns": final_df.columns.tolist(),
            "records": final_df.head(10).values.tolist(),
            "total_size": len(final_df),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        return render(request, 'admin/preprocess.html', context)

    # ---------------------------------------------------------
    # 2️⃣ If no preprocessed data → preprocess raw files
    # ---------------------------------------------------------

    all_files = [f for f in os.listdir(raw_folder) if f.endswith(".csv")]

    if len(all_files) == 0:
        return render(request, "admin/preprocess.html", {
            "msg": "❌ No CSV files found in raw dataset folder!"
        })

    preprocessed_list = []

    for file in all_files:
        file_path = os.path.join(raw_folder, file)
        df = pd.read_csv(file_path, low_memory=False)

        df.columns = df.columns.str.strip().str.lower()
        available_cols = [col for col in selected_norm if col in df.columns]
        df = df[available_cols]

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        if "label" in df.columns:
            df["label"] = df["label"].apply(lambda x: 0 if str(x).upper() == "BENIGN" else 1)

        df.to_csv(os.path.join(preprocessed_folder, file), index=False)
        preprocessed_list.append(df)

    final_df = pd.concat(preprocessed_list, ignore_index=True)

    X = final_df.drop("label", axis=1)
    y = final_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save model training datasets
    if not os.path.exists("model"):
        os.makedirs("model")

    joblib.dump(X_train, "model/X_train.pkl")
    joblib.dump(X_test, "model/X_test.pkl")
    joblib.dump(y_train, "model/y_train.pkl")
    joblib.dump(y_test, "model/y_test.pkl")

    context = {
        "msg": "✅ Preprocessing Completed Successfully!",
        "columns": final_df.columns.tolist(),
        "records": final_df.head(10).values.tolist(),
        "total_size": len(final_df),
        "train_size": len(X_train),
        "test_size": len(X_test)
    }

    return render(request, 'admin/preprocess.html', context)



def build_model(request):

    MODEL_DIR = "model"

    # ---------------------------
    # Helpers
    # ---------------------------
    def save_fig(fig, path):
        fig.savefig(path, format="png", bbox_inches="tight")
        plt.close(fig)

    def img_to_base64(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # ---------------------------
    # Load dataset
    # ---------------------------
    try:
        X_train = joblib.load(os.path.join(MODEL_DIR, "X_train.pkl"))
        X_test = joblib.load(os.path.join(MODEL_DIR, "X_test.pkl"))
        y_train = joblib.load(os.path.join(MODEL_DIR, "y_train.pkl"))
        y_test = joblib.load(os.path.join(MODEL_DIR, "y_test.pkl"))

    except Exception as e:
        print("❌ Dataset loading error:", e)
        traceback.print_exc()  # prints full stack trace in terminal

        return render(request, "admin/build_model.html", {
            "msg": f"❌ Dataset loading error: {e}"
        })

    # ---------------------------
    # File paths
    # ---------------------------
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    dnn_path = os.path.join(MODEL_DIR, "dnn_model.h5")

    gan_gen_path = os.path.join(MODEL_DIR, "gan_generator.h5")
    gan_disc_path = os.path.join(MODEL_DIR, "gan_discriminator.h5")
    gan_aug_model_path = os.path.join(MODEL_DIR, "gan_augmented_rf.pkl")
    gan_metrics_path = os.path.join(MODEL_DIR, "gan_metrics.json")

    # ============================================================
    # CASE 1: RF & DNN exist, BUT GAN NOT TRAINED → Train GAN only
    # ============================================================
    if os.path.exists(metrics_path) and os.path.exists(rf_path) and os.path.exists(dnn_path):
        
        metrics = json.load(open(metrics_path))
        gan_exists = os.path.exists(gan_metrics_path)

        if not gan_exists:
            # =========================
            # TRAIN GAN NOW
            # =========================
            input_dim = X_train.shape[1]

            # Generator
            generator = Sequential([
                Input(shape=(input_dim,)),
                Dense(64, activation="relu"),
                Dense(input_dim, activation="linear")
            ])
            generator.compile(optimizer=Adam(0.001), loss="mse")

            # Discriminator
            discriminator = Sequential([
                Input(shape=(input_dim,)),
                Dense(64, activation="relu"),
                Dense(1, activation="sigmoid")
            ])
            discriminator.compile(optimizer=Adam(0.001), loss="binary_crossentropy")

            # Combined GAN
            discriminator.trainable = False
            gan = Sequential([generator, discriminator])
            gan.compile(optimizer=Adam(0.001), loss="binary_crossentropy")

            minority = X_train[y_train == 1].values
            steps = 200

            for step in range(steps):
                noise = np.random.normal(0, 1, minority.shape)
                fake = generator.predict(noise, verbose=0)

                discriminator.trainable = True
                discriminator.train_on_batch(minority, np.ones((len(minority), 1)))
                discriminator.train_on_batch(fake, np.zeros((len(fake), 1)))

                discriminator.trainable = False
                gan.train_on_batch(noise, np.ones((len(fake), 1)))

            # Save GAN models
            generator.save(gan_gen_path)
            discriminator.save(gan_disc_path)

            # Generate synthetic data
            synthetic = generator.predict(np.random.normal(0, 1, (500, input_dim)), verbose=0)
            y_synth = np.ones(500)

            X_aug = np.vstack([X_train.values, synthetic])
            y_aug = np.hstack([y_train.values, y_synth])

            rf_aug = RandomForestClassifier(n_estimators=150, random_state=42)
            rf_aug.fit(X_aug, y_aug)

            y_pred_aug = rf_aug.predict(X_test)
            accuracy_aug = round(accuracy_score(y_test, y_pred_aug) * 100, 2)

            joblib.dump(rf_aug, gan_aug_model_path)

            # Confusion matrix for augmented model
            cm_aug = confusion_matrix(y_test, y_pred_aug)
            fig = plt.figure(figsize=(4, 4))
            plt.imshow(cm_aug, cmap="Greens")
            plt.title("GAN-Augmented Confusion Matrix")
            for i in range(cm_aug.shape[0]):
                for j in range(cm_aug.shape[1]):
                    plt.text(j, i, cm_aug[i, j], ha="center", va="center")
            save_fig(fig, MODEL_DIR + "confusion_matrix_aug.png")

            json.dump({"accuracy_augmented": accuracy_aug}, open(gan_metrics_path, "w"))

            # Return after GAN training
            return render(request, "admin/build_model.html", {
                "msg": "✔ RF & DNN Loaded<br>✔ GAN Trained Successfully",

                "accuracy_rf": metrics["accuracy_rf"],
                "accuracy_dnn": metrics["accuracy_dnn"],
                "classification_report": metrics["classification_report"],

                "roc_curve_graph": img_to_base64(MODEL_DIR + "roc_curve.png"),
                "confusion_matrix_graph": img_to_base64(MODEL_DIR + "confusion_matrix.png"),
                "feature_importance_graph": img_to_base64(MODEL_DIR + "feature_importance.png"),
                "shap_plot_graph": img_to_base64(MODEL_DIR + "shap_summary.png"),
                "lime_plot_graph": img_to_base64(MODEL_DIR + "lime_plot.png"),

                "gan_status": "✔ GAN Trained",
                "accuracy_augmented": accuracy_aug,
                "confusion_matrix_aug_graph": img_to_base64(MODEL_DIR + "confusion_matrix_aug.png"),
            })

        # All models exist — just load & show
        return render(request, "admin/build_model.html", {
            "msg": "✔ Models Built Successfully...!",

            "accuracy_rf": metrics["accuracy_rf"],
            "accuracy_dnn": metrics["accuracy_dnn"],
            "classification_report": metrics["classification_report"],

            "roc_curve_graph": img_to_base64(MODEL_DIR + "roc_curve.png"),
            "confusion_matrix_graph": img_to_base64(MODEL_DIR + "confusion_matrix.png"),
            "feature_importance_graph": img_to_base64(MODEL_DIR + "feature_importance.png"),
            "shap_plot_graph": img_to_base64(MODEL_DIR + "shap_summary.png"),
            "lime_plot_graph": img_to_base64(MODEL_DIR + "lime_plot.png"),

            "gan_status": "✔ GAN Trained" if gan_exists else "❌ GAN Not Trained",
            "accuracy_augmented": json.load(open(gan_metrics_path))["accuracy_augmented"] if gan_exists else None,
            "confusion_matrix_aug_graph": img_to_base64(MODEL_DIR + "confusion_matrix_aug.png") if gan_exists else None,
        })

    # ============================================================
    # CASE 2: NO MODELS EXIST → TRAIN RF + DNN + GAN FROM SCRATCH
    # ============================================================
    msg = []

    # ---------------- RF ----------------
    rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    accuracy_rf = round(accuracy_score(y_test, y_pred) * 100, 2)
    class_report_str = classification_report(y_test, y_pred)

    joblib.dump(rf, rf_path)
    msg.append("✔ Random Forest Trained")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    save_fig(fig, MODEL_DIR + "confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig = plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    save_fig(fig, MODEL_DIR + "roc_curve.png")

    # Feature importance
    fig = plt.figure(figsize=(6, 5))
    plt.barh(X_train.columns, rf.feature_importances_)
    save_fig(fig, MODEL_DIR + "feature_importance.png")

    # ---------------- DNN ----------------
    dnn = Sequential([
        Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dropout(0.20),
        Dense(1, activation="sigmoid"),
    ])
    dnn.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
    dnn.fit(X_train.values, y_train.values, epochs=6, batch_size=64, verbose=0)

    dnn.save(dnn_path)
    msg.append("✔ DNN Trained")

    dnn_eval = dnn.evaluate(X_test.values, y_test.values, verbose=0)
    accuracy_dnn = round(dnn_eval[1] * 100, 2)

    # ---------------- SHAP ----------------
    try:
        subset = X_train.sample(150)
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(subset)

        fig = plt.figure(figsize=(7, 5))
        shap.summary_plot(shap_vals, subset, show=False)
        save_fig(fig, MODEL_DIR + "shap_summary.png")
        msg.append("✔ SHAP Saved")
    except:
        msg.append("❌ SHAP Failed")

    # ---------------- LIME ----------------
    try:
        def dnn_predict(x):
            p = dnn.predict(x)
            p = p.reshape(-1, 1)
            return np.hstack([1-p, p])

        expl = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=["Benign", "Attack"],
            mode="classification",
        )

        exp = expl.explain_instance(
            X_test.iloc[0].values,
            dnn_predict,
            num_features=6
        )

        fig = exp.as_pyplot_figure()
        save_fig(fig, MODEL_DIR + "lime_plot.png")
        msg.append("✔ LIME Saved")
    except:
        msg.append("❌ LIME Failed")

    # ---------------- GAN Training ----------------
    input_dim = X_train.shape[1]

    generator = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(input_dim, activation="linear")
    ])
    generator.compile(optimizer=Adam(0.001), loss="mse")

    discriminator = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    discriminator.compile(optimizer=Adam(0.001), loss="binary_crossentropy")

    discriminator.trainable = False
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(0.001), loss="binary_crossentropy")

    minority = X_train[y_train == 1].values
    steps = 200

    for step in range(steps):
        noise = np.random.normal(0, 1, minority.shape)
        fake = generator.predict(noise, verbose=0)

        discriminator.trainable = True
        discriminator.train_on_batch(minority, np.ones((len(minority), 1)))
        discriminator.train_on_batch(fake, np.zeros((len(fake), 1)))

        discriminator.trainable = False
        gan.train_on_batch(noise, np.ones((len(fake), 1)))

    generator.save(gan_gen_path)
    discriminator.save(gan_disc_path)
    msg.append("✔ GAN Trained")

    synthetic = generator.predict(np.random.normal(0, 1, (500, input_dim)), verbose=0)
    y_synth = np.ones(500)

    X_aug = np.vstack([X_train.values, synthetic])
    y_aug = np.hstack([y_train.values, y_synth])

    rf_aug = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_aug.fit(X_aug, y_aug)

    y_pred_aug = rf_aug.predict(X_test)
    accuracy_aug = round(accuracy_score(y_test, y_pred_aug) * 100, 2)
    joblib.dump(rf_aug, gan_aug_model_path)

    # Confusion matrix for augmented model
    cm_aug = confusion_matrix(y_test, y_pred_aug)
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm_aug, cmap="Greens")
    for i in range(cm_aug.shape[0]):
        for j in range(cm_aug.shape[1]):
            plt.text(j, i, cm_aug[i, j], ha="center", va="center")
    save_fig(fig, MODEL_DIR + "confusion_matrix_aug.png")

    json.dump({"accuracy_augmented": accuracy_aug}, open(gan_metrics_path, "w"))

    # Save main metrics
    metrics = {
        "accuracy_rf": accuracy_rf,
        "accuracy_dnn": accuracy_dnn,
        "classification_report": class_report_str
    }
    json.dump(metrics, open(metrics_path, "w"))

    # ---------------- Return everything ----------------
    return render(request, "admin/build_model.html", {
        "msg": "<br>".join(msg),

        "accuracy_rf": accuracy_rf,
        "accuracy_dnn": accuracy_dnn,
        "classification_report": class_report_str,

        "roc_curve_graph": img_to_base64(MODEL_DIR + "roc_curve.png"),
        "confusion_matrix_graph": img_to_base64(MODEL_DIR + "confusion_matrix.png"),
        "feature_importance_graph": img_to_base64(MODEL_DIR + "feature_importance.png"),
        "shap_plot_graph": img_to_base64(MODEL_DIR + "shap_summary.png"),
        "lime_plot_graph": img_to_base64(MODEL_DIR + "lime_plot.png"),

        "gan_status": "✔ GAN Trained",
        "accuracy_augmented": accuracy_aug,
        "confusion_matrix_aug_graph": img_to_base64(MODEL_DIR + "confusion_matrix_aug.png"),
    })




def user_registration(request):
    return render(request,'user/user_registration.html')


def user_registration_action(request):
    username = request.POST.get('username')
    email = request.POST.get('email')
    password = request.POST.get('password')
    confirm_password = request.POST.get('confirm_password')
    if password != confirm_password:
        return render(request,'user/user_registration.html',{'msg':'Password and Confirm password did not match..!'})
    con = pymysql.connect(
        host = 'localhost',
        user = 'root',
        password = 'root',
        database = 'cyber_attack_prediction',
        charset = 'utf8'
    )
    cur = con.cursor()
    cur.execute("SELECT * from users where username=%s or email=%s",(username,email))
    existing_user = cur.fetchone()
    if existing_user:
        con.close()
        return render(request,'user/user_registration.html',{'msg':'Username or Email already exists...!'})
    cur.execute(
        "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
        (username, email, password)
    )
    con.commit()
    con.close()

    return render(request, 'user/user_registration.html', {'msg': 'Registration Successful...! Please Login...!'})


def user_login(request):
    return render(request,'user/user_login.html')

def user_login_action(request):
    username = request.POST.get('username')
    password = request.POST.get('password')
    con = pymysql.connect(host="localhost", user="root", password="root", database="cyber_attack_prediction", charset='utf8')
    cur = con.cursor()

    cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
    user = cur.fetchone()
    con.close()

    if user:
        return render(request, 'user/user_home.html', {'username': username})
    else:
        return render(request, 'user/user_login.html', {'msg': 'Invalid username or password'})
    

def user_home(request):
    return render(request, 'user/user_home.html')




def enter_test_data(request):

    MODEL_DIR = "model/"
    rf_model         = joblib.load(MODEL_DIR + "rf_model.pkl")
    dnn_model        = load_model(MODEL_DIR + "dnn_model.h5")
    rf_gan_augmented = joblib.load(MODEL_DIR + "gan_augmented_rf.pkl")

    X_train          = joblib.load(MODEL_DIR + "X_train.pkl")
    X_test           = joblib.load(MODEL_DIR + "X_test.pkl")

    # Load Models
    try:
        rf_model         = joblib.load(MODEL_DIR + "rf_model.pkl")
        dnn_model        = load_model(MODEL_DIR + "dnn_model.h5")
        rf_gan_augmented = joblib.load(MODEL_DIR + "gan_augmented_rf.pkl")

        X_train          = joblib.load(MODEL_DIR + "X_train.pkl")
        X_test           = joblib.load(MODEL_DIR + "X_test.pkl")
    except Exception as e:
        print("Loading Models Error: "+e)
        return render(request, "user/enter_test_data.html", {
            "msg": "❌ Model files missing. Please ask Admin to train the model first."
        })

    # If Form Submitted
    if request.method == "POST":

        try:
            fields = [
                "flow_duration", "total_fwd_packets", "total_backward_packets",
                "total_length_fwd", "total_length_bwd", "flow_bytes",
                "flow_packets", "flow_iat_mean", "fwd_packet_length_mean",
                "packet_length_std", "min_packet", "max_packet",
                "syn_flag", "ack_flag"
            ]

            data = {f: float(request.POST.get(f)) for f in fields}

        except:
            return render(request, "user/enter_test_data.html", {
                "msg": "❌ Please enter valid numeric values."
            })

        # Convert to DataFrame (same as training)
        df_input = pd.DataFrame({
            "flow duration":              [data["flow_duration"]],
            "total fwd packets":          [data["total_fwd_packets"]],
            "total backward packets":     [data["total_backward_packets"]],
            "total length of fwd packets":[data["total_length_fwd"]],
            "total length of bwd packets":[data["total_length_bwd"]],
            "flow bytes/s":               [data["flow_bytes"]],
            "flow packets/s":             [data["flow_packets"]],
            "flow iat mean":              [data["flow_iat_mean"]],
            "fwd packet length mean":     [data["fwd_packet_length_mean"]],
            "packet length std":          [data["packet_length_std"]],
            "min packet length":          [data["min_packet"]],
            "max packet length":          [data["max_packet"]],
            "syn flag count":             [data["syn_flag"]],
            "ack flag count":             [data["ack_flag"]]
        })

        # --- Predictions ---
        rf_pred = int(rf_model.predict(df_input)[0])
        dnn_prob = dnn_model.predict(df_input.values).flatten()[0]
        dnn_pred = 1 if dnn_prob >= 0.5 else 0
        gan_pred = int(rf_gan_augmented.predict(df_input)[0])

        votes = [rf_pred, dnn_pred, gan_pred]
        final = max(set(votes), key=votes.count)

        def label(x):
            return "🚨 ATTACK DETECTED" if x == 1 else "✅ BENIGN TRAFFIC"

        # Text Results
        rf_text  = label(rf_pred)
        dnn_text = label(dnn_pred)
        gan_text = label(gan_pred)
        result_label = label(final)

        # -----------------------------------------
        # SHAP PLOT (Bar Chart – Fully working)
        # -----------------------------------------
        shap_plot_path = MODEL_DIR + "shap_output.png"
        shap_base64 = None

        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(df_input)

            plt.figure(figsize=(8, 5))
            shap.bar_plot(
                shap_values[1][0],
                feature_names=df_input.columns,
                show=False
            )
            plt.tight_layout()
            plt.savefig(shap_plot_path)
            plt.close()

            with open(shap_plot_path, "rb") as f:
                shap_base64 = base64.b64encode(f.read()).decode()

        except Exception as e:
            print("SHAP ERROR:", e)
            shap_base64 = None

        # -----------------------------------------
        # LIME PLOT (Works as-is)
        # -----------------------------------------
        lime_plot_path = MODEL_DIR + "lime_output.png"
        lime_base64 = None

        try:
            def dnn_predict(x):
                p = dnn_model.predict(x)
                p = p.reshape(-1, 1)
                return np.hstack([1 - p, p])

            lime_explainer = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=["Benign", "Attack"],
                mode="classification"
            )

            lime_exp = lime_explainer.explain_instance(
                df_input.values[0],
                dnn_predict,
                num_features=6
            )

            fig = lime_exp.as_pyplot_figure()
            fig.savefig(lime_plot_path, bbox_inches="tight")
            plt.close(fig)

            with open(lime_plot_path, "rb") as f:
                lime_base64 = base64.b64encode(f.read()).decode()

        except Exception as e:
            print("LIME ERROR:", e)
            lime_base64 = None

        # Return to template
        return render(request, "user/enter_test_data.html", {
            "msg": "Prediction Completed Successfully!",
            "result": result_label,
            "rf_pred": rf_text,
            "dnn_pred": dnn_text,
            "gan_pred": gan_text,
            "shap_plot": shap_base64,
            "lime_plot": lime_base64
        })

    return render(request, "user/enter_test_data.html")




sns.set(style="whitegrid")


# ------------------------------------------
# Save Figure to Disk
# ------------------------------------------
def save_fig(fig, path):
    fig.savefig(path, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)


# ------------------------------------------
# Clean filename
# ------------------------------------------
def clean_filename(name):
    return name.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")


def analysis_graphs(request):

    # Paths
    BASE_DIR = settings.BASE_DIR
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "cyber_attack")
    SAVE_DIR = os.path.join(BASE_DIR, "static", "analysis_graphs")

    # Create folder if not exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Step 1: If images already exist → just load them
    saved_images = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith(".png")])

    if saved_images:
        image_paths = [f"analysis_graphs/{img}" for img in saved_images]
        return render(request, "user/analysis_graphs.html", {"images": image_paths})

    # Step 2: Load dataset
    if not os.path.exists(DATASET_PATH):
        return render(request, "user/analysis_graphs.html", {"error": "Dataset folder missing"})

    csv_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".csv")]

    if not csv_files:
        return render(request, "user/analysis_graphs.html", {"error": "No CSV files found"})

    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(os.path.join(DATASET_PATH, f)))
        except:
            continue

    df = pd.concat(df_list, ignore_index=True)
    df.columns = df.columns.str.lower().str.strip()

    selected_cols = [
        "flow duration", "total fwd packets", "total backward packets",
        "total length of fwd packets", "total length of bwd packets",
        "flow bytes/s", "flow packets/s", "flow iat mean",
        "fwd packet length mean", "packet length std",
        "min packet length", "max packet length",
        "syn flag count", "ack flag count",
        "label"
    ]

    df = df[selected_cols]

    df["label"] = df["label"].apply(lambda x: "Attack" if str(x).upper() != "BENIGN" else "Benign")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Remove infinite values
    df[numeric_cols] = df[numeric_cols].replace([float("inf"), float("-inf")], pd.NA)

    # Cap extreme values
    for col in numeric_cols:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    img_index = 1  # for saving unique file names

    # -------------------------------------------------
    # 1. Label Distribution
    # -------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    df["label"].value_counts().plot(kind="bar", color=["green", "red"])
    plt.title("Attack vs Benign Distribution")
    save_fig(fig, os.path.join(SAVE_DIR, f"{img_index}_label_distribution.png"))
    img_index += 1

    # -------------------------------------------------
    # 2. Histograms
    # -------------------------------------------------
    for col in numeric_cols:
        fig = plt.figure(figsize=(6, 4))
        df[col].plot(kind="hist", bins=30)
        plt.title(f"Distribution: {col}")
        filename = f"{img_index}_hist_{clean_filename(col)}.png"
        save_fig(fig, os.path.join(SAVE_DIR, filename))
        img_index += 1

    # -------------------------------------------------
    # 3. Boxplots
    # -------------------------------------------------
    for col in numeric_cols:
        fig = plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
        filename = f"{img_index}_box_{clean_filename(col)}.png"
        save_fig(fig, os.path.join(SAVE_DIR, filename))
        img_index += 1

    # -------------------------------------------------
    # 4. Correlation Heatmap
    # -------------------------------------------------
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    save_fig(fig, os.path.join(SAVE_DIR, f"{img_index}_corr_heatmap.png"))
    img_index += 1

    # -------------------------------------------------
    # 5. Mean Comparison
    # -------------------------------------------------
    for col in numeric_cols:
        fig = plt.figure(figsize=(6, 4))
        df.groupby("label")[col].mean().plot(kind="bar")
        plt.title(f"Mean Comparison of {col}")
        filename = f"{img_index}_mean_{clean_filename(col)}.png"
        save_fig(fig, os.path.join(SAVE_DIR, filename))
        img_index += 1

    # -------------------------------------------------
    # 6. Scatter Plots
    # -------------------------------------------------
    pairs = [
        ("flow duration", "flow bytes/s"),
        ("total fwd packets", "total backward packets"),
        ("min packet length", "max packet length")
    ]

    for x, y in pairs:
        fig = plt.figure(figsize=(6, 4))
        sns.scatterplot(x=df[x], y=df[y], hue=df["label"])
        plt.title(f"{x} vs {y}")
        filename = f"{img_index}_scatter_{clean_filename(x)}_{clean_filename(y)}.png"
        save_fig(fig, os.path.join(SAVE_DIR, filename))
        img_index += 1

    # -----------------------------------------
    # Load all saved images for rendering
    # -----------------------------------------
    saved_images = sorted([f"analysis_graphs/{f}" for f in os.listdir(SAVE_DIR) if f.endswith(".png")])

    return render(request, "user/analysis_graphs.html", {"images": saved_images})

