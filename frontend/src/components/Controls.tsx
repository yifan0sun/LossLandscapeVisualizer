import React from 'react';
import Plot from 'react-plotly.js';


 

// two plots 
export function Controls({ 
    handleReset, handlePlay, isPlaying, 
    selectedZoom,setSelectedZoom,
    selectedEpoch, setSelectedEpoch
    }: any) {
    return (
        

 
<div style={{ 
    marginTop: "0px", 
    display: "flex", 
    flexDirection: "row",    // ✅ row instead of column
    alignItems: "top",    // ✅ center items vertically
    gap: "20px",             // ✅ gap between controls
    padding: "20px",
    flexWrap: "wrap",         // ✅ allow wrapping if window shrinks
    marginBottom:"100px",
    flexShrink: 0         // ✅ Prevent it from shrinking
  }}>
  

 
    <div style={{ display: "flex", gap: "12px" }}>
      <button onClick={handleReset} style={{ padding: "6px 12px" }}>Reset</button>
      <button onClick={handlePlay} style={{ padding: "6px 12px" }}>
        {isPlaying ? "Stop" : "Play"}
      </button>
    </div>

    {/* Zoom Slider */}
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
    <label htmlFor="zoom-slider">Zoom: {selectedZoom}</label>
      <input
        id="zoom-slider"
        type="range"
        min={0}
        max={2}
        step={1}
        value={["0.01", "0.1", "1"].indexOf(selectedZoom)}
        onChange={(e) => {
          const options = ["0.01", "0.1", "1"];
          const idx = parseInt(e.target.value);
          setSelectedZoom(options[idx]);
        }}
        style={{ width: "300px" }}
      />
    </div>

    
    {/* Epoch Slider */}
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <label htmlFor="epoch-slider" style={{ whiteSpace: "nowrap" }}>Epoch: {selectedEpoch}</label>
      <input
        id="epoch-slider"
        type="range"
        min={1}
        max={1000}
        step={1}
        value={selectedEpoch }
        onChange={(e) => setSelectedEpoch(parseInt(e.target.value) )}
        style={{ width: "300px" }}
      />
    </div>



  </div>

  
);
}