# backend/server.py
from flask import Flask, jsonify,send_from_directory, request
import os
from flask_cors import CORS
import pickle
import numpy as np
import torch

app = Flask(__name__)
CORS(app)  # allow React to make requests

ROOT_DIR = os.path.dirname(__file__)
#MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "previews")
LANDSCAPE_DIR = os.path.join(os.path.dirname(__file__), "landscapes")
DB_DIR = os.path.join(os.path.dirname(__file__), "decisionboundaries")
"""
@app.route("/list_models")
def list_models():
    model_dirs = []
    for entry in os.scandir(MODELS_DIR):
        if entry.is_dir():
            if entry.name.startswith("mlp_"):
                # Remove "mlp_" prefix and split the numbers
                numbers = entry.name.replace("mlp_", "").split("_")
                formatted = "[" + ", ".join(numbers) + "]"
                model_dirs.append(formatted)
            else:
                model_dirs.append(entry.name)  # fallback, if no mlp_ prefix
    return jsonify(model_dirs)
 



@app.route("/selected_model/<model>")
def selected_model(model):
    print(f"✅ Model selected on frontend: {model}")
    return '', 204  # No content
"""

@app.route("/list_datasets")
def list_datasets():
    print("🔵 /list_datasets was called")
    datasets = set()
    for entry in os.scandir(DATA_DIR):
        if entry.is_file():
            name = entry.name
            if name.endswith(".png"):
                base = name.replace(".png", "")
                datasets.add(base)
    print(datasets)
    return jsonify(sorted(datasets))



@app.route("/selected_dataset/<dataset>")
def selected_dataset(dataset):
    print(f"✅ dataset selected on frontend: {dataset}")
    return '', 204  # No content

@app.route("/static/previews/<filename>")
def serve_preview(filename): 
    print(f"✅ preview selected on frontend: {DATA_DIR},{filename}")

    return send_from_directory(DATA_DIR, filename)



@app.route("/selection", methods=["POST"])
def selection():
    data = request.get_json()
    print("✅ Selection received:")
    print(f"Model: {data.get('model')}")
    print(f"Dataset: {data.get('dataset')}")
    print(f"Epoch: {data.get('epoch')}")
    print(f"Zoom: {data.get('zoom')}")
    print(f"Split: {data.get('split')}")
    return '', 204  # no content


@app.route("/get_surface_plot_data", methods=["POST"])
def get_surface_plot_data():
    data = request.get_json()

    model = data.get('model')  
    dataset = data.get('dataset')
    epoch = data.get('epoch')
    zoom = data.get('zoom')
    split = 'train'



    # Build path 
    cache_land_dir = os.path.join(
        LANDSCAPE_DIR,
        model,
        dataset,
        split,
        f"range{zoom}",
        f"ep{epoch}"
    ) 
    cache_range_dir = os.path.join(
        LANDSCAPE_DIR,
        model,
        dataset,
        split,
        f"range{zoom}"
    )
     
    print(cache_land_dir,cache_range_dir)

    
    a_path = os.path.join(cache_land_dir, "a_vals.npy")
    b_path = os.path.join(cache_land_dir, "b_vals.npy")
    loss_path = os.path.join(cache_land_dir, "loss.npy")
    zrange_path = os.path.join(cache_range_dir, "zrange.npy")
 
    if not (os.path.exists(a_path) and os.path.exists(b_path) and os.path.exists(loss_path) and os.path.exists(zrange_path)):
        return jsonify({"error": "Data not found"}), 404
    
    a_vals = np.load(a_path)
    b_vals = np.load(b_path)
    losses = np.load(loss_path)
    zrange = np.load(zrange_path)
    
    output = {}
    output['a'] = a_vals.tolist()
    output['b'] = b_vals.tolist()
    output['loss'] = losses.tolist()
    output['zrange'] = zrange.tolist()
    return jsonify(output)





@app.route("/get_decbnd_plot_data", methods=["POST"])
def get_decbnd_plot_data():
    data = request.get_json()

    model = data.get('model')  
    dataset = data.get('dataset')
    epoch = data.get('epoch')


    # Build path
    
    cache_bound_dir = os.path.join(
        DB_DIR,
        model,
        dataset
    )
    print(cache_bound_dir)
    
    decision_output_dir = os.path.join(cache_bound_dir, "ep%d.pkl" % epoch)
 
    
    if not (os.path.exists(decision_output_dir)):
        return jsonify({"error": "Data not found"}), 404

    output = pickle.load(open(decision_output_dir,'rb'))
    output['x'] = output['xx'][:,0]
    output['y'] = output['yy'][0,:] 
    output['preds'] = output['preds'].T

    for k,v in output.items():
        if k == 'train_labels' or k == 'test_labels':
            output[k] = output[k][:,0].tolist() 
        else:
            output[k] = output[k].tolist() 
            
    return jsonify(output)

if __name__ == "__main__":
    #app.run(debug=True, port=5000)
    app.run(host="0.0.0.0", port=10000)