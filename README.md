 
# Loss Landscape and Decision Boundary Visualizer

A web-based tool for exploring neural network loss landscapes and decision boundaries during training.

---

## 🌟 Overview

This tool allows users to interactively visualize:

- **Loss landscapes** — how a neural network's loss behaves near its trained parameters.
- **Decision boundaries** — how the trained model separates input space into predicted classes.

By exploring these visualizations, users can gain intuition about model optimization, dataset difficulty, and learning dynamics.

---

## 📊 Visualizations

There are two plots shown in the center panel:

### 1. Loss Landscape (Left Plot)

- Shows the **loss surface** around the model's trained parameters.
- Plotted in 3D using two random perturbation directions (newly generated models with random instantiation)

### 2. Decision Boundary (Right Plot)

- Shows the **predicted class regions** for the input space.
- The background color represents model prediction over a 2D grid.
- Data points are shown as **red or blue dots**, where color corresponds to the true class label.


---

## 🧭 How to Use the App

1. **Select a model** from the left panel (e.g., number and size of hidden layers).
2. **Select a dataset** from the right panel.
3. **Adjust training epoch** with the Epoch slider to explore different training stages.
4. **Adjust zoom range** around the trained parameters if desired.
5. **Click Play** to animate the landscape through epochs.
6. **Reset** to return to initial view.

---
 

## 📎 Technical Notes

- **Loss surfaces** are precomputed for a grid of `a`, `b` directions and stored as NumPy arrays.
- **Decision boundaries** are generated using model predictions over a 2D meshgrid.
- **Plots** are rendered with Plotly.js for full interactivity (rotation, zooming disabled manually).

✅ Users **do not** need to install anything — the app runs in-browser.

---


## 📬 Questions, ideas, or feedback?  
Email: `yifan dot zero dot sun at gmail dot com`  
Website: [optimalvisualizer.com](http://optimalvisualizer.com)
 