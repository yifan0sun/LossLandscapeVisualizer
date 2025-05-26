import React, { useState, useEffect ,startTransition } from "react";
//import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

import './App.css';
import { useRef } from "react";

 
import { Plots } from './components/Plots';
import { Controls } from './components/Controls';
import { LeftPanel } from "./components/LeftPanel";
import {AboutPage} from "./components/about"
//const BASE_URL = "http://localhost:5000";
const BASE_URL = "https://losslandscapevisualizer.onrender.com";




function App() {

const [page, setPage] = useState<'main' | 'about'>('main');
 

  // Selection states
//const [selectedModel, setSelectedModel] = useState<string | null>(null);
const [selectedDataset, setSelectedDataset] = useState<string | null>(null);
const [selectedEpoch, setSelectedEpoch] = useState<number>(0);
const [selectedZoom, setSelectedZoom] = useState<string>("1");


const [selectedWidth, setSelectedWidth] = useState<number>(10);
const [selectedDepth, setSelectedDepth] = useState<number>(2);
// Data
//const [modelFiles, setModelFiles] = useState<string[]>([]);
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

// Playback
const [isPlaying, setIsPlaying] = useState(false);
const playInterval = useRef<NodeJS.Timeout | null>(null);



const [frozenZRange, setFrozenZRange] = useState<[number, number] | null>(null);
const [scaleType, setScaleType] = useState("linear"); // 🔹 this is new

 
const handleReset = () => {
  setCamera(null);
  setSelectedEpoch(0);
  setFrozenZRange(null); //
};

const handlePlay = () => {
  if (isPlaying) {
    if (playInterval.current) {
      clearInterval(playInterval.current);
      playInterval.current = null;
    }
    setIsPlaying(false);
    setFrozenZRange(null);
  } else {
    setIsPlaying(true);
    if (surfaceData?.zrange) {
      setFrozenZRange(surfaceData.zrange);
    }

    let currentEpoch = selectedEpoch;

    const interval = setInterval(() => {
      currentEpoch += 10;
      if (currentEpoch > 1000) {
        clearInterval(interval);
        playInterval.current = null;
        setIsPlaying(false);
        setFrozenZRange(null);
      } else {
        startTransition(() => {
          setSelectedEpoch(currentEpoch);
        });
      }
    }, 300);

    playInterval.current = interval;
  }
};




///////////////////////////////////////////////////////////////////

 

  useEffect(() => {
    fetch(`${BASE_URL}/list_datasets`)
      .then(res => res.json())
      .then(data => {
        setDatasetFiles(data);
        if (data.length > 0) {
          setSelectedDataset(data[0]); // ✅ Automatically select the first dataset
        }
      });
  }, []);

  useEffect(() => {
    if (!selectedDataset) return;
  
    const archArray = Array(selectedDepth).fill(selectedWidth);
    const modelName = "mlp_" + archArray.join("_");
  
    const payload = {
      model: modelName,
      dataset: selectedDataset,
      epoch: selectedEpoch,
      zoom: selectedZoom
    };
  
    fetch(`${BASE_URL}/get_surface_plot_data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data) {
          setSurfaceData(data);
          setFrozenZRange(data.zrange);
        }
      });
  
    fetch(`${BASE_URL}/get_decbnd_plot_data`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data) {
          setDecbndData(data);
        }
      });
  
  }, [selectedDataset, selectedDepth, selectedWidth, selectedZoom, selectedEpoch]);
  

  if (page === 'about') {
      return <AboutPage onBack={() => setPage('main')} />;
    }
 

  return (

    

    <div className="app-container">



      
      {/* Left Panel */}

     
       <LeftPanel
       setSelectedDepth={setSelectedDepth}
       setSelectedWidth={setSelectedWidth}
       selectedDepth={selectedDepth}
       selectedWidth={selectedWidth}
       /> 

      {/* Center Panel */}
      <div className="center-panel"> 
      <div className="center-scrollable">
          
        <p style={{ textAlign: "left" , marginBottom: "0px" }}><b>About.</b> This tool visualizes loss landscapes during neural network training, providing insight
         into model behavior near local optima. By observing the deformation of the optimization surface close to the trained model (at (0,0)), we can better understand why some models are
          easier or harder to train — for example, due to the presence of sharp minima or flat regions. Note that landscape characteristics depend on many factors,  e.g., class separability, geometry of loss function, etc.   </p>
          <p>

   <button
  onClick={() => setPage('about')}
style={{
    padding: '2px 8px',
    width: 'fit-content',
    maxWidth: '200px',
    whiteSpace: 'nowrap'
  }}
>Click for more details</button>
</p>

        <p style={{ textAlign: "left" , marginBottom: "0px" }}><b>More.</b> Visit  <a href="https://github.com/yifan0sun/LossLandscapeVisualizer/blob/main/README.md">the project README file</a></p>
        <p style={{ textAlign: "left" , marginBottom: "0px" }}><b>Prototype.</b>  Feedback and suggestions are welcome! Please visit <a href="https://sites.google.com/view/visualizerprojects/home">optimalvisualizer.com</a> to give feedback or visit more visually pleasing explainability apps.</p>
          
                  
      <Plots
          frozenZRange={frozenZRange}
          surfaceData={surfaceData}
          decbndData={decbndData}
          camera={camera}
          setCamera={setCamera}
          scaleType={scaleType}
          selectedDataset={selectedDataset}
          selectedEpoch={selectedEpoch}
          selectedZoom={selectedZoom} 
          selectedDepth={selectedDepth}
          selectedWidth={selectedWidth}
        />


  


      <Controls
          handleReset={handleReset}
          handlePlay={handlePlay}
          isPlaying={isPlaying}
          selectedZoom={selectedZoom}
          setSelectedZoom={setSelectedZoom}
          selectedEpoch={selectedEpoch}
          setSelectedEpoch={setSelectedEpoch}
          scaleType={scaleType}
          setScaleType={setScaleType}
        /> 


</div>

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
                src={`${BASE_URL}/static/previews/${dataset}.png`}
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
