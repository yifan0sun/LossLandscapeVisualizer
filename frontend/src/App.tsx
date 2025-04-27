import React, { useState, useEffect } from "react";
import Plot from 'react-plotly.js';
import './App.css';
import { useRef } from "react";

 
import { Plots } from './components/Plots';
import { Controls } from './components/Controls';


 


function App() {



  // Selection states
const [selectedModel, setSelectedModel] = useState<string | null>(null);
const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
const [selectedEpoch, setSelectedEpoch] = useState<number>(1);
const [selectedZoom, setSelectedZoom] = useState<string>("0.1");
//const [selectedSplit, setSelectedSplit] = useState<"train" | "test">("train");
const selectedSplit = 'train'
const [selectedWidth, setSelectedWidth] = useState<number>(10);
const [selectedDepth, setSelectedDepth] = useState<number>(2);
// Data
const [modelFiles, setModelFiles] = useState<string[]>([]);
const [datasetFiles, setDatasetFiles] = useState<string[]>([]);

const [surfaceData, setSurfaceData] = useState<{
  a: number[],
  b: number[],
  loss: number[][],
  zrange: [number, number],
} | null>(null);


const [decbndData, setDecbndData] = useState<{
  x: number[],
  y: number[],
  xx: number[][],
  yy: number[][],
  preds: number[][],
  train_inputs: number[][],
  train_labels: number[],
  test_inputs: number[][],
  test_labels: number[],
} | null>(null);


// Visualization
const [camera, setCamera] = useState<any>(null);
const [zRange, setZRange] = useState<[number, number] | null>(null);

// Playback
const [isPlaying, setIsPlaying] = useState(false);
const playInterval = useRef<NodeJS.Timeout | null>(null);



const [frozenZRange, setFrozenZRange] = useState<[number, number] | null>(null);


 
const handleReset = () => {
  setSelectedZoom("1.0");
  setCamera(null);
  setSelectedEpoch(1);
  setZRange(null); // ✅ optionally reset fixed zRange
};



const handlePlay = () => {
  if (isPlaying) {
    // If already playing, then STOP
    if (playInterval.current) {
      clearInterval(playInterval.current);
      playInterval.current = null;
    }
    setIsPlaying(false);
    setFrozenZRange(null); // ✅ Clear frozenZRange on stop
  } else {
    // Start playing
    setIsPlaying(true);    
    if (surfaceData?.zrange) {
      setFrozenZRange(surfaceData.zrange); // ✅ Freeze the current zrange
    }
    let currentEpoch = selectedEpoch;  // ✅ start from wherever it currently is

 

    const interval = setInterval(() => {
      currentEpoch += 10;
      if (currentEpoch > 1000) {
        clearInterval(interval);
        playInterval.current = null;
        setIsPlaying(false);
        setFrozenZRange(null);
      } else {
        setSelectedEpoch(currentEpoch);
      }
    }, 300);

    playInterval.current = interval;
  }
}; 




///////////////////////////////////////////////////////////////////





  useEffect(() => {
    fetch("http://localhost:5000/list_models")
      .then(res => res.json())
      .then(data => {
        setModelFiles(data);
        if (data.length > 0) {
          setSelectedModel(data[0]); // ✅ Automatically select the first model
        }
      });
  }, []);

  useEffect(() => {
    fetch("http://localhost:5000/list_datasets")
      .then(res => res.json())
      .then(data => {
        setDatasetFiles(data);
        if (data.length > 0) {
          setSelectedDataset(data[0]); // ✅ Automatically select the first dataset
        }
      });
  }, []);


  useEffect(() => {
    const arch = Array(selectedDepth).fill(selectedWidth);  // e.g., [10, 10] for width=10, depth=2
    const archName = "mlp_" + arch.join("_");
    setSelectedModel(archName);
  }, [selectedWidth, selectedDepth]);


  useEffect(() => {
    if (!selectedModel || !selectedDataset || selectedEpoch === null) return; // ✅ block if missing
    const archArray = Array(selectedDepth).fill(selectedWidth); // [10,10] or [64,64,64]
    const modelName = "mlp_" + archArray.join("_"); // "mlp_10_10" or "mlp_64_64_64"

  
    const payload = {
      model: modelName,
      dataset: selectedDataset,
      epoch: selectedEpoch,
      zoom: selectedZoom,
      split: selectedSplit
    };
  
    // 1. Send to /selection (logging)
    fetch("http://localhost:5000/selection", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => {
        if (!res.ok) {
          console.warn("Failed to send selection");
        }
      })
      .catch((err) => {
        console.error("Fetch error (selection):", err);
      });
  
    // 2. Send to /get_surface_plot_data (actual plot)
    fetch("http://localhost:5000/get_surface_plot_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => {
        if (!res.ok) {
          console.warn("Surface data not found");
          return null;
        }
        return res.json();
      })
      .then(data => {
        if (data && data.a && data.b && data.loss && data.zrange) {
          setSurfaceData(data);
        } else {
          setSurfaceData(null);  // ✅ Important: clear surfaceData if incomplete!
          console.warn("Incomplete surface data received:", data);
        }
      })
      .catch(err => {
        console.error("Fetch error (surface data):", err);
      });

      
    // 2. Send to /get_decbnd_plot_data (actual plot)
    fetch("http://localhost:5000/get_decbnd_plot_data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => {
        if (!res.ok) {
          console.warn("Decision boundary data not found");
          return null;
        }
        return res.json();
      })
      .then(data => {
        if (data && data.x && data.y && data.xx && data.yy && data.preds) {
          setDecbndData(data);
        } else {
          setDecbndData(null);  // ✅ Important: clear surfaceData if incomplete!
          console.warn("Incomplete decision boundary data received:", data);
        }
      })
      .catch(err => {
        console.error("Fetch error (decision boundary data):", err);
      });
  
  }, [selectedWidth, selectedDepth, selectedDataset, selectedEpoch, selectedZoom, selectedSplit]);
  



  




  return (
    <div className="app-container">
      
      {/* Left Panel */}
      <div className="left-panel">

        <h3>Pick a model</h3>
        <p>Select a multilayer perceptron architecture. Input: 2D. Output: BCE loss.</p>


        <div style={{
          width: "100%",
          height: "2px",
          backgroundColor: "#cccccc",
          margin: "16px 0"
        }} />


        <p><b>Model</b></p>
        <div style={{ marginBottom: "10px" }}>
          <label>Width:</label>
          <select value={selectedWidth} onChange={(e) => setSelectedWidth(Number(e.target.value))}>
            {[10, 25,50,100].map(width => (
              <option key={width} value={width}>{width}</option>
            ))}
          </select>
        </div>

        <div style={{ marginBottom: "20px" }}>
          <label>Depth:</label>
          <select value={selectedDepth} onChange={(e) => setSelectedDepth(Number(e.target.value))}>
            {[1, 2, 3, 4].map(depth => (
              <option key={depth} value={depth}>{depth}</option>
            ))}
          </select>
        </div>

        <div style={{
          width: "100%",
          height: "2px",
          backgroundColor: "#cccccc",
          margin: "16px 0"
        }} />
        
        <div className="model-cartoon">
          <div style={{ textAlign: "center", fontSize: "14px", color: "#333" }}>
            <div>Output</div>

            {/* Bars */}
            {Array(selectedDepth).fill(selectedWidth).map((width, idx) => (
              <React.Fragment key={idx}>
                <div
                  style={{
                    height: "12px",
                    width: `${50 + width / 5}px`,
                    backgroundColor: "#4db6ac",
                    margin: "12px auto",
                    borderRadius: "4px",
                  }}
                ></div> 
              </React.Fragment>
            ))}

            <div>Input</div>
          </div>
        </div>

      </div>

      {/* Center Panel */}
      <div className="center-panel"> 
          
                  <p style={{ textAlign: "left" , marginBottom: "0px" }}><b>About.</b> This tool visualizes loss landscapes during neural network training, providing insight into model behavior near local optima. By observing the deformation of the optimization surface, we can better understand why some models are easier or harder to train — for example, due to the presence of sharp minima or flat regions. Note that landscape characteristics depend on many factors, including dataset hardness (e.g., class separability, geometry).</p>
          
                  <p style={{ textAlign: "left" , marginBottom: "0px" }}><b>More.</b> Visit  <a href="https://github.com/yifan0sun/LossLandscapeVisualizer/blob/main/README.md">the project README file</a></p>
                  <p style={{ textAlign: "left" , marginBottom: "0px" }}><b>Prototype.</b> This is an early version of the tool. Feedback and suggestions are welcome! Please contact me at "yifan dot sun@stonybrook dot edu" or visit <a href="http://optimalvisualizer.com/">optimalvisualizer.com</a> for more visually pleasing explainability apps.</p>
          
                  
      <Plots
          surfaceData={surfaceData}
          decbndData={decbndData}
          camera={camera}
          setCamera={setCamera}
        />

      <Controls
          handleReset={handleReset}
          handlePlay={handlePlay}
          isPlaying={isPlaying}
          selectedZoom={selectedZoom}
          setSelectedZoom={setSelectedZoom}
          selectedEpoch={selectedEpoch}
          setSelectedEpoch={setSelectedEpoch}
        /> 



      </div>

      {/* Right Panel */}
      <div className="right-panel">
        <h3>Pick a dataset</h3>
        
        <div className="dataset-list">
          {datasetFiles.map((dataset) => (
            <div
              key={dataset}
              className={`dataset-folder ${selectedDataset === dataset ? "selected" : ""}`}
              onClick={() => {
                setSelectedDataset(dataset);
                console.log(`✅ Dataset selected: ${dataset}`);
              }}
            >
              <img
                src={`http://localhost:5000/static/previews/${dataset}.png`}
                alt={dataset}
                style={{
                  width: "75%",            // ✅ smaller width
                  display: "block",         // ✅ block-level element
                  margin: "10px auto",      // ✅ center it horizontally
                  borderRadius: "8px",      // ✅ optional: rounded corners
                  backgroundColor: "transparent", // ✅ no white border background
                  padding: "0",             // ✅ no padding
              }}
              />
               
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}

export default App;
