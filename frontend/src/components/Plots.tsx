import React from 'react';
import Plot from 'react-plotly.js';


 

// two plots 
export function Plots({ surfaceData, decbndData, camera, setCamera, 
    selectedDataset,  selectedEpoch, selectedZoom, selectedSplit, selectedDepth,selectedWidth
    }: any) {
    return (
        
 
        <div className="plots-container">
        <SurfacePlot
            surfaceData={surfaceData}
            camera={camera}
            setCamera={setCamera}
            selectedDataset={selectedDataset}
            selectedEpoch={selectedEpoch}
            selectedZoom={selectedZoom}
            selectedSplit={selectedSplit}
            selectedDepth={selectedDepth}
            selectedWidth={selectedWidth}
        />
      
        <DecbndPlot
            decbndData={decbndData}
            camera={camera}
            setCamera={setCamera}
            selectedDataset={selectedDataset}
            selectedEpoch={selectedEpoch}
            selectedDepth={selectedDepth}
            selectedWidth={selectedWidth}
        />
        </div> 
    );
}

// Plotly surface plot 
export function SurfacePlot({ surfaceData, camera, setCamera,  
    selectedDataset,  selectedEpoch, selectedZoom, selectedSplit, selectedDepth,selectedWidth
    }: any) {
    return (

        
        
        <div className="single-plot">
            {surfaceData && surfaceData.a && surfaceData.b && surfaceData.loss && surfaceData.zrange ? (
                <Plot
                style={{ width: "100%", height: "100%" }}
                data={[
                    {
                    type: 'surface',
                    x: surfaceData.a,
                    y: surfaceData.b,
                    z: surfaceData.loss,
                    colorscale: 'Viridis',
                    showscale: false,    // ✅ ADD THIS
                    colorbar: { show: false }
                    }
                ]}
                layout={{
                    title: `Loss Landscape`,
                    autosize: true,
                    width: undefined,  // ✅ Let width flex
                    height: undefined, // ✅ No hardcoded height
                    scene: {
                    camera: camera || {},  // ✅ Use saved camera, or default if not available
                    aspectmode: 'manual',  // ✅ Very important
                    aspectratio: { x: 1, y: 1, z: 0.7 },  // ✅ Fix x:y:z ratios manually
                    xaxis: { title: 'a' },
                    yaxis: { title: 'b' },
                    zaxis: {
                        title: 'Loss',
                        range: surfaceData?.zrange || undefined 
                    }
                    },
                    uirevision: 'static', // ✅ This preserves zoom/pan/rotation exactly
                    margin: { l: 0, r: 0, t: 30, b: 0 }
                }}
                config={{
                    responsive: true,
                    scrollZoom: false,
                    useResizeHandler: true  // ✅ ADD THIS

                }}
                onRelayout={(figure: any) => {
                    if (figure['scene.camera']) {
                    setCamera(figure['scene.camera']);  // ✅ Capture latest camera on interaction
                    }
                }}
                />
            ) : (

            <div style={{ padding: "20px", fontSize: "16px", color: "#555" }}>
                <p>⚠️ Data not yet available.</p>
                <p><strong>Details of missing data:</strong></p>
                <ul>
                <li>Model: {"mlp_" + Array(selectedDepth).fill(selectedWidth).join("_")}</li>
                <li>Dataset: {selectedDataset || "(none selected)"}</li>
                <li>Epoch: {selectedEpoch}</li>
                <li>Zoom: {selectedZoom}</li>
                <li>Split: {selectedSplit}</li>
                </ul>
            </div>
            )}
        </div>
    );
}


 

//Decision boundary plot 

export function DecbndPlot({ decbndData, camera, setCamera,
        selectedDataset,  selectedEpoch,   selectedDepth,selectedWidth
    }: any) {
    return (




    
    

    <div className="single-plot">
        {decbndData && decbndData.x && decbndData.y && decbndData.xx && decbndData.yy && decbndData.preds ? (

        <Plot
        style={{ width: "100%", height: "100%" }}
        data={[
            // Background decision boundary contour
            {
            type: 'contour',
            x: decbndData.x,                // ✅ x-values are from first row
            y: decbndData.y, // ✅ y-values are from first column
            z: decbndData.preds,                 // ✅ full preds grid
            
            showscale: false,
            colorscale: "RdBu",
            contours: {
                coloring: "heatmap",
                showlines: true,
                colorbar: { show: false },
                smoothing: 1.3
            },
            line: {
                width: 1  // ✅ Optional: line thickness, you can adjust this too
            },
            opacity: 0.8
            },
            // Training points
            {
            type: 'scatter',
            mode: 'markers',
            x: decbndData.train_inputs.map((row: number[]) => row[0]),
            y: decbndData.train_inputs.map((row: number[]) => row[1]),
            marker: {
                color: decbndData.train_labels.map((label: number) => label === 1 ? 'red' : 'blue'),
                colorscale: "RdBu",
                line: { color: "black", width: .5 },
                size: 6
            },
            name: "Train",
            showlegend: false,    // ✅ Disable legend for train
            },
        ]}
        layout={{
            title: "Decision Boundary",
            autosize: true,
            width: undefined,
            height: undefined, // ✅ No hardcoded height
            xaxis: {
                showgrid: false,        // ✅ No background grid
                zeroline: false,        // ✅ No bold line at 0
                showline: false,        // ✅ No axis border
                showticklabels: false,  // ✅ No numbers
                ticks: '',              // ✅ No ticks
                title: 'x',
                scaleanchor: 'y',    // ✅ Lock x and y scales
                scaleratio: 1,
                constrain: 'domain', // ✅ Make sure x uses its domain without stretching
            },
            yaxis: {
                showgrid: false,
                zeroline: false,
                showline: false,
                showticklabels: false,
                title: 'y',
                constrain: 'domain', // ✅ Make y fill the domain square
            },
            margin: { l: 40, r: 20, t: 30, b: 40 },
            dragmode: false
        }}
        config={{
            responsive: true,
            scrollZoom: false,
            useResizeHandler: true  // ✅ ADD THIS            
        }}
        />
        ) : (
        <div style={{ padding: "20px", fontSize: "16px", color: "#555" }}>
        <p>⚠️ Decision boundary data not yet available.</p>
        <div style={{ padding: "20px", fontSize: "16px", color: "#555" }}>
                <p>⚠️ Data not yet available.</p>
                <p><strong>Details of missing data:</strong></p>
                <ul>
                <li>Model: {"mlp_" + Array(selectedDepth).fill(selectedWidth).join("_")}</li>
                <li>Dataset: {selectedDataset || "(none selected)"}</li>
                <li>Epoch: {selectedEpoch}</li>
                </ul>
            </div>
        </div>
        )}
    </div> 
    
);
}
