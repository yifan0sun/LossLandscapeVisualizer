import Plot from 'react-plotly.js';
import React, { useRef,useState, useEffect } from "react";
 


 

// two plots 
export const Plots = React.memo( function Plots({frozenZRange,  surfaceData, decbndData, camera, setCamera, 
    selectedDataset,  selectedEpoch, selectedZoom,  selectedDepth,selectedWidth,scaleType
    }: any) {


        //const [zRange, setZRange] = useState<[number, number] | null>(null);

    

        

    return (
        
<div className="plots-container" >

        
<div className="left-plot"  >
       
        <SurfacePlot
            frozenZRange={frozenZRange}
            surfaceData={surfaceData}
            camera={camera}
            setCamera={setCamera}
            selectedDataset={selectedDataset}
            selectedEpoch={selectedEpoch}
            selectedZoom={selectedZoom}
            selectedDepth={selectedDepth}
            selectedWidth={selectedWidth}
            scaleType={scaleType}
        />
</div>
<div className="right-plot"  >

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
</div>
    );
});
 



// Plotly surface plot 
export const SurfacePlot = React.memo(function SurfacePlot({ frozenZRange,    surfaceData, camera, setCamera,  
    selectedDataset,  selectedEpoch, selectedZoom,  selectedDepth,selectedWidth,
    scaleType
    }: any) { 
 

        const containerRef = useRef<HTMLDivElement>(null);
        const [plotDims, setPlotDims] = useState<{ width: number; height: number } | null>(null);

        useEffect(() => {
            const updateSize = () => {
              if (containerRef.current) {
                const { offsetWidth, offsetHeight } = containerRef.current;
                if (offsetWidth > 0 && offsetHeight > 0) {
                  setPlotDims({ width: offsetWidth, height: offsetHeight });
                }
              }
            };
          
            window.addEventListener("resize", updateSize);
          
            // ✅ Immediate layout check (works on most platforms)
            updateSize();
          
            // ✅ Force second check after layout stabilizes (for iPad/Safari)
            const retry = setTimeout(() => {
              updateSize();
            }, 1000);
          
            return () => {
              clearTimeout(retry);
              window.removeEventListener("resize", updateSize);
            };
          }, []);
          


        if (!surfaceData || !surfaceData.loss || !surfaceData.a || !surfaceData.b || !surfaceData.zrange) {
            return <div style={{ padding: "20px", fontSize: "16px", color: "#555" }}>⚠️ Waiting for surface data...</div>;
          }
          
          const processedLoss = scaleType === "log"
            ? surfaceData.loss.map((row: number[]) =>
                row.map((val: number) => Math.log10(Math.max(val, 1e-8))))
            : surfaceData.loss;
          
          const transformedZRange = (() => {
            if (scaleType === "log") {
              const [zmin, zmax] = surfaceData.zrange;
              const safeMin = Math.log10(Math.min(Math.max(zmin, 1e-5), .1));
              const safeMax = Math.log10(Math.max(zmax, 1e-8)) + Math.min(1., .1 * (Math.log10(zmax) - Math.log10(zmin)));
              return [safeMin, safeMax];
            } else {
              return surfaceData.zrange;
            }
          })();

        
      
    return (

        
<div
  ref={containerRef}
  style={{
    width: "100%",
    height: "100%",
    maxWidth: "100%",
    maxHeight: "100%",
    overflow: "hidden",
    boxSizing: "border-box"
  }}
>
  {
   surfaceData && surfaceData.a && surfaceData.b && surfaceData.loss && surfaceData.zrange ? (
 


                <Plot
                style={{ width: "100%", height: "100%" }}
                useResizeHandler={true}
                data={[
                    {
                    type: 'surface',
                    x: surfaceData.a,
                    y: surfaceData.b,
                    z: processedLoss,
                    colorscale: 'Viridis',
                    showscale: false,    // ✅ ADD THIS
                    colorbar: { show: false }
                    }
                ]}
                
                layout={{ 
                    title: `Loss Landscape`, 
                    width: plotDims?.width,
                    height: plotDims?.height,
                    autosize: false, 
                    scene: {
                        ...(camera ? { camera } : {}),
                        aspectmode: 'manual',
                        aspectratio: { x: 1, y: 1, z: 0.7 },
                        xaxis: {
                            title: { text: 'perturbation direction 1', font: { size: 12 } },
                            showticklabels: true,
                            ticks: '',
                            ticklen: 0
                        },
                        yaxis: {
                            title: { text: 'perturbation direction 1', font: { size: 12 } },
                            showticklabels: true,
                            ticks: '',
                            ticklen: 0
                        },
                        zaxis: {
                            title: {text:  scaleType === "log" ? 'loss (log)' : 'loss', font: { size: 12 } },
                            range: transformedZRange,
                            showticklabels: true,
                            ticks: '',
                            ticklen: 0
                        },
                    },
                    uirevision: 'static', //  
                    margin: { l: 0, r: 0, t: 30, b: 0 }
                }} 
                config={{
                    responsive: false,
                    scrollZoom: false,
                    useResizeHandler: false  // ✅ ADD THIS

                }}
                onRelayout={(figure: any) => {
                    const newCamera = figure['scene.camera'];
                    if (
                      newCamera &&
                      JSON.stringify(newCamera) !== JSON.stringify(camera)
                    ) {
                      setCamera(newCamera);  // ✅ Only update if truly changed
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
                </ul>
            </div>
)}  
</div>
    );
});

//Decision boundary plot 

export const DecbndPlot = React.memo(function DecbndPlot({ decbndData, camera, setCamera,
        selectedDataset,  selectedEpoch,   selectedDepth,selectedWidth
    }: any) {
    return (




    
    

<div style={{ width: "100%", maxWidth: "100%", height: "100%" }}>
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
                title: {
                    text: 'feature 1',   // 🔥 set dummy
                    font: { size: 14 },       // 🔥 force title font size
                },
                scaleanchor: 'y',    // ✅ Lock x and y scales
                scaleratio: 1,
                constrain: 'domain', // ✅ Make sure x uses its domain without stretching
            },
            yaxis: {
                showgrid: false,
                zeroline: false,
                showline: false,
                showticklabels: false,
                ticks: '',   // ✅ No ticks
                title: {
                    text: 'feature 2',   // 🔥 set dummy
                    font: { size: 14 },       // 🔥 force title font size
                },
                constrain: 'domain', // ✅ Make y fill the domain square
            },
            margin: { l: 50, r: 0, t: 30, b: 40 },
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
});
