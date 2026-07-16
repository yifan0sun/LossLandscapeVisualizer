import React, { useState, startTransition } from 'react';


 

// two plots 
export const Controls = React.memo(function Controls({ 
    handleReset, handlePlay, isPlaying, 
    selectedZoom,setSelectedZoom,
    selectedEpoch, setSelectedEpoch,
    scaleType, setScaleType
    }: any) {

      const [tempEpoch, setTempEpoch] = useState(selectedEpoch);
      const zoomOptions = ["0.1", "0.2", "0.5", "1", "2", "5", "10"];
      const [tempZoomIndex, setTempZoomIndex] = useState(
        zoomOptions.indexOf(selectedZoom)
      );
    return (
        

 
<div style={{ 
    marginTop: "0px", 
    display: "flex", 
    flexDirection: "row",    // ✅ row instead of column
    alignItems: "top",    // ✅ center items vertically
    gap: "20px",             // ✅ gap between controls
    padding: "20px",
    flexWrap: "wrap",         // ✅ allow wrapping if window shrinks
    marginBottom:"10px",
    flexShrink: 0,         // ✅ Prevent it from shrinking
    flex: "0 0 auto", // ✅ do not shrink or stretch
    minWidth: "100%", // ✅ full width so it doesn’t get squished
  }}>
  

 {/* Scale Selector */}
<div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
  <label>
    <input
      type="radio"
      value="linear"
      checked={scaleType === "linear"}
      onChange={() => setScaleType("linear")}
    />
    Linear
  </label>
  <label>
    <input
      type="radio"
      value="log"
      checked={scaleType === "log"}
      onChange={() => setScaleType("log")}
    />
    Log
  </label>
</div>


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
        max={6}
        step={1}
        value={["0.1", "0.2", "0.5", "1", "2", "5", "10"].indexOf(selectedZoom)}
        onChange={(e) => {
          const idx = parseInt(e.target.value);
          const newZoom = ["0.1", "0.2", "0.5", "1", "2", "5", "10"][idx];
          startTransition(() => setSelectedZoom(newZoom));
        }}
         
        //onChange={(e) => {
       //   const options = [ "0.1","0.2","0.5", "1","2","5", "10"];
        //  const idx = parseInt(e.target.value); 
        //  startTransition(() => {
        //    setSelectedZoom(options[idx]);
        //  })
       // }}
        style={{ width: "300px" }}
      />
    </div>

    
    {/* Epoch Slider */}
    <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
      <label htmlFor="epoch-slider" style={{ whiteSpace: "nowrap" }}>Epoch: {selectedEpoch}</label>
      <input
        id="epoch-slider"
        type="range"
        min={0}
        max={1000}
        step={10}
        value={selectedEpoch}
        onChange={(e) => {
          const newEpoch = parseInt(e.target.value);
          startTransition(() => setSelectedEpoch(newEpoch));
        }}
        
        style={{ width: "300px" }}
      />
    </div>



  </div>

  
);
});