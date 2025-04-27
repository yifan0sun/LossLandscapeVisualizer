# backend/server.py
from flask import Flask, jsonify
import os
from flask_cors import CORS

import numpy as np
import torch

app = Flask(__name__)
CORS(app)  # allow React to make requests

MODELS_ROOT = os.path.join(os.path.dirname(__file__), "../models")

@app.route("/list_models")
def list_models():
    model_paths = []
    for root, _, files in os.walk(MODELS_ROOT):
        for f in files:
            if f.endswith(".pth"):
                rel_path = os.path.relpath(os.path.join(root, f), MODELS_ROOT)
                model_paths.append(rel_path.replace("\\", "/"))  # normalize Windows paths
    return jsonify(model_paths)
 



@app.route("/get_surface_data/<model>/<epoch>/<ab_range>")
def get_surface_data(model, epoch,ab_range):
    
    model_id = f"{model}"
    base_path = os.path.join("..", "landscapes", model_id, "random_instance_1", f"ep{epoch}", f"range{ab_range}")

    a_path = os.path.join(base_path, "a_vals.npy")
    b_path = os.path.join(base_path, "b_vals.npy")
    loss_path = os.path.join(base_path, "loss.npy")

    if not os.path.exists(a_path):
        return jsonify({"error": "Missing a_vals.npy"}), 404
    if not os.path.exists(b_path):
        return jsonify({"error": "Missing b_vals.npy"}), 404
    if not os.path.exists(loss_path):
        return jsonify({"error": "Missing loss.npy"}), 404
     
    try:
        a_vals = np.load(a_path).tolist()
        b_vals = np.load(b_path).tolist()
        loss = np.load(loss_path).tolist()

        json_obj = jsonify({
            "a": a_vals,
            "b": b_vals,
            "loss": loss
        }) 
        return json_obj
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)