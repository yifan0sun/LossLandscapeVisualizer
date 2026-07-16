import React from "react";
//import Plot from 'react-plotly.js';



 
export function   LeftPanel({ setSelectedDepth, selectedDepth,setSelectedWidth, selectedWidth
    }: any) {

 
 


  return (

      <div className="left-panel">

        <h3>Pick a model</h3>
        <p>Select a multilayer perceptron architecture.</p>


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
            {[5, 10, 25,50,100].map(width => (
              <option key={width} value={width}>{width}</option>
            ))}
          </select>
        </div>

        <div style={{ marginBottom: "20px" }}>
          <label>Depth:</label>
          <select value={selectedDepth} onChange={(e) => setSelectedDepth(Number(e.target.value))}>
            {[1,2,4,8,16].map(depth => (
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
  );
}
 
