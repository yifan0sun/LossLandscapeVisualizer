

# Loss Landscape and Decision Boundary Visualizer

A web-based tool for exploring neural network loss landscapes and decision boundaries during training.

---

## 🌟 Overview

This tool allows users to interactively visualize:

- **Loss landscapes** — how a neural network's loss behaves near its trained parameters.
- **Decision boundaries** — how the trained model separates input space into predicted classes.

By exploring these visualizations, users can gain intuition about model optimization, dataset difficulty, and learning dynamics.

---

## 🖥️ Application Layout

The app is divided into three main panels:

| Panel | Purpose |
|:------|:--------|
| **Left Panel** | Select the model architecture (e.g., number and width of hidden layers). |
| **Center Panel** | View visualizations: loss surface and decision boundary. Control epoch, zoom, and animation. |
| **Right Panel** | Select the dataset used for training/testing. |

✅ The interface is responsive:  
- Wide screens display plots side-by-side  
- Narrow screens stack plots vertically

✅ Controls (Reset, Play, Epoch slider, Zoom slider) always stay at the bottom.

---

## 📊 Visualizations

There are two plots shown in the center panel:

### 1. Loss Landscape (Left Plot)

- Shows the **loss surface** around the model's trained parameters.
- Plotted in 3D using two random perturbation directions (`a`, `b`).
- Helps you see whether the optimizer reached a sharp minimum, wide basin, saddle point, etc.
- Smoother surfaces generally indicate better generalization.

### 2. Decision Boundary (Right Plot)

- Shows the **predicted class regions** for the input space.
- The background color represents model prediction over a 2D grid.
- Training points are shown as **red or blue dots**.
- Testing points are shown as **red or blue X's**.
- Color corresponds to the true class label.

✅ Axes are locked to equal aspect ratio to preserve true geometry.

---

## 🧭 How to Use the App

1. **Select a model** from the left panel (e.g., number and size of hidden layers).
2. **Select a dataset** from the right panel.
3. **Adjust training epoch** with the Epoch slider to explore different training stages.
4. **Adjust zoom range** around the trained parameters if desired.
5. **Click Play** to animate the landscape through epochs.
6. **Reset** to return to initial view.

---

## 🛠️ Controls

| Control | Purpose |
|:--------|:--------|
| **Reset** | Reset zoom and rotation to defaults. |
| **Play / Stop** | Animate epochs in sequence. |
| **Epoch Slider** | Jump to a specific training epoch (in steps). |
| **Zoom Slider** | Adjust zoom-in or zoom-out level around optimum. |

✅ All controls update both plots automatically.

---

## 📎 Technical Notes

- **Loss surfaces** are precomputed for a grid of `a`, `b` directions and stored as NumPy arrays.
- **Decision boundaries** are generated using model predictions over a 2D meshgrid.
- **Plots** are rendered with Plotly.js for full interactivity (rotation, zooming disabled manually).

✅ Users **do not** need to install anything — the app runs in-browser.

---

## 🧪 Prototype Disclaimer

This tool is an early prototype.

- Some UI features and plots are under active development.
- Suggestions for improvement are welcome!

---

## 📬 Contact

- Built by **Yifan Sun**.
- Questions, ideas, or feedback?  
Email: `yifan dot sun at stonybrook dot edu`  
Website: [optimalvisualizer.com](http://optimalvisualizer.com)

---

# 📋 Final Notes

✅ Please fill in:
- (optional) Any known limitations  
- (optional) Planned future features (e.g., different loss functions? different architectures?)

✅ Otherwise, this README is fully aligned with your app.

 