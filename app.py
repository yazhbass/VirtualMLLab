from flask import Flask, render_template, abort
import subprocess
import os
import sys

app = Flask(__name__)

# ==========================================================
# Experiment List
# ==========================================================

experiments = [
    {"id": 1, "title": "Linear Regression on Diabetes Dataset"},
    {"id": 2, "title": "Logistic Regression Classification"},
    {"id": 3, "title": "Support Vector Machine (SVM) Classification"},
    {"id": 4, "title": "K-Nearest Neighbor (KNN) Classification"},
    {"id": 5, "title": "Decision Tree Classification"},
    {"id": 6, "title": "K-Means Clustering"},
    {"id": 7, "title": "Random Forest Classification"},
    {"id": 8, "title": "AdaBoost Classification"},
    {"id": 9, "title": "Hierarchical Clustering"},
    {"id": 10, "title": "Naive Bayes Classification"},
    {"id": 11, "title": "Convolutional Neural Network (CNN)"}
]

# ==========================================================
# Graph Mapping
# ==========================================================

graph_files = {

    1: {"graph": "graphs/regression.png"},

    2: {"graph": "graphs/logistic.png"},

    3: {"graph": "graphs/svm.png"},

    4: {"graph": "graphs/knn.png"},

    5: {
        "graph1": "graphs/iris_confusion_matrix.png",
        "graph2": "graphs/iris_decision_tree.png"
    },

    6: {"graph": "graphs/kmeans.png"},

    7: {
        "graph1": "graphs/random_forest.png",
        "graph2": "graphs/random_forest_features.png"
    },

    8: {"graph": "graphs/adaboost.png"},

    9: {"graph": "graphs/hierarchical.png"},

    10: {"graph": "graphs/naive_bayes.png"},

    11: {"graph": "graphs/cnn_accuracy.png"}

}

# ==========================================================
# Create Graph Folder Automatically
# ==========================================================

os.makedirs("static/graphs", exist_ok=True)

# ==========================================================
# Welcome Page
# ==========================================================

@app.route("/")
def welcome():
    return render_template("welcome.html")

# ==========================================================
# Dashboard
# ==========================================================

@app.route("/home")
def home():
    return render_template(
        "index.html",
        experiments=experiments
    )

# ==========================================================
# Open Experiment
# ==========================================================

@app.route("/experiment/<int:id>")
def experiment(id):

    if id < 1 or id > 11:
        abort(404)

    graphs = graph_files.get(id, {})

    return render_template(
        f"exp{id}.html",
        experiment=experiments[id - 1],
        output=None,
        graph=graphs.get("graph"),
        graph1=graphs.get("graph1"),
        graph2=graphs.get("graph2"),
        graph3=graphs.get("graph3")
    )

# ==========================================================
# Run Experiment
# ==========================================================

@app.route("/run_experiment/<int:id>")
def run_experiment(id):

    if id < 1 or id > 11:
        abort(404)

    script = os.path.join(
        "experiments",
        f"exp{id}.py"
    )

    output = ""

    if os.path.isfile(script):

        try:

            result = subprocess.run(
                [sys.executable, script],
                capture_output=True,
                text=True,
                timeout=120
            )

            output = result.stdout

            if result.stderr:
                output += "\n\nERROR:\n"
                output += result.stderr

        except subprocess.TimeoutExpired:
            output = "Experiment execution timed out."

        except Exception as e:
            output = str(e)

    else:
        output = f"{script} not found."

    graphs = graph_files.get(id, {})

    return render_template(
        f"exp{id}.html",
        experiment=experiments[id - 1],
        output=output,
        graph=graphs.get("graph"),
        graph1=graphs.get("graph1"),
        graph2=graphs.get("graph2"),
        graph3=graphs.get("graph3")
    )

# ==========================================================
# Error Page
# ==========================================================

@app.errorhandler(404)
def page_not_found(error):
    return "<h2>404 - Page Not Found</h2>", 404

# ==========================================================
# Run Flask
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)