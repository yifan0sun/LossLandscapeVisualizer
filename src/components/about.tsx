export function AboutPage({ onBack }: { onBack: () => void }) {
  return (
    <div style={{ padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <h2 style={{ marginBottom: '0.5rem', fontSize: '1.75rem' }}>Loss Landscape Visualizers</h2>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>Motivation</h3>
        <p style={paragraphStyle}>
          Visualizing the loss landscape of a neural network is a way to better understand how training behaves. It can help explain why some architectures are easier to train than others, how optimization moves through parameter space, and how difficult the task is. This type of visualization can also give insights into common machine learning challenges like overfitting, oversmoothing, continual learning, or multimodal objectives.
        </p>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>How this tool works</h3>
        <p style={paragraphStyle}>
          We pick two random directions in weight space — each defined by a randomly initialized model — and compute the loss as we move along those directions from the trained model. This creates a 2D "slice" of the full high-dimensional loss surface, helping us see how flat, sharp, or irregular the region is around the solution. In this simulation, we use Xavier (Glorot) initialization for those random directions. To present a more comprehensive view, the tool offers limited zooming (we need to save datasets per zoom level, and the storage does not scale well), and the trajectory along the training.
        </p>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>Why MLPs?</h3>
        <p style={paragraphStyle}>
          MLPs (multi-layer perceptrons) are used because they are easy to define and modify. We can adjust the number of layers (depth) and the number of neurons per layer (width) and see how this changes training behavior and the resulting loss landscape.
        </p>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>The datasets</h3>
        <p style={paragraphStyle}>
          All datasets here are 2D classification tasks. This makes it possible to both plot the decision boundary and understand how hard the task is visually. Some datasets, like spirals, are more difficult to learn because they require complex boundaries. Others, like overlapping blobs, may seem easy but can still be overfit if the model is too flexible. You can explore how task complexity and architecture interact.
        </p>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>What you can explore</h3>
        <p style={paragraphStyle}>
          By changing the depth and width of the model, you can see how optimization changes. Some architectures fall into sharp valleys, while others land in flatter regions. These differences can show how easy or hard it is for gradient descent to find a good solution, and whether that solution generalizes well or not.
        </p>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>Limitations</h3>
        <p style={paragraphStyle}>
          This kind of loss visualization has limitations. The directions we use are random, so different choices may produce different-looking landscapes. A flat valley in one slice might look sharp in another. Also, these directions might not align with the most meaningful directions for training. In practice, better tools like gradient or Hessian-based analyses can give more targeted insights, but are less visually intuitive. 
        </p>
      </div>

      <div style={cardStyle}>
        <h3 style={sectionTitleStyle}>About me</h3>
        <p style={paragraphStyle}>
          I'm <a href="https://optimalyifansun.com/" target="_blank" rel="noopener noreferrer">Yifan Sun</a>, from Stony Brook University. My goal in creating this app — and other <a href="https://www.optimalvisualizer.com/" target="_blank" rel="noopener noreferrer">interactive visual tools</a> — is to help connect theoretical ideas in machine learning with practical, intuitive understanding. I believe that when explanations are grounded in simple, visual examples, abstract ideas become much more accessible. This project is one step toward bridging that gap.
        </p>
      </div>

      <button onClick={onBack} style={{ alignSelf: 'flex-start', marginTop: '1rem' }}>← Back to main</button>
    </div>
  );
}

const cardStyle: React.CSSProperties = {
  backgroundColor: '#f9f9f9',
  border: '1px solid #ddd',
  borderRadius: '8px',
  padding: '1rem',
  boxShadow: '0 2px 4px rgba(0,0,0,0.05)'
};

const sectionTitleStyle: React.CSSProperties = {
  fontSize: '1.1rem',
  fontWeight: 600,
  marginBottom: '0.5rem'
};

const paragraphStyle: React.CSSProperties = {
  fontSize: '0.95rem',
  lineHeight: 1.6
};