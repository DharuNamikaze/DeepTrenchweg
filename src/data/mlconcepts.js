// src/data/mlConcepts.js

export const mlConcepts = [
  // === SURFACE ZONE (0-1000m): Fundamentals ===
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    depth: 150,
    image: 'linearReg.webp',
    description: 'Finding the best-fit line through data points',
    category: 'supervised-learning',
    size: 'large'
  },
  {
    id: 'logistic-regression',
    name: 'Logistic Regression',
    depth: 400,
    image: 'logistic-regression.webp',
    description: 'Classification using the sigmoid function',
    category: 'supervised-learning',
    size: 'medium'
  },
  {
    id: 'decision-tree',
    name: 'Decision Tree',
    depth: 700,
    image: 'decision-tree.webp',
    description: 'Tree-based decisions for classification',
    category: 'supervised-learning',
    size: 'medium'
  },
  
  // === TWILIGHT ZONE (1000-3000m): Neural Networks ===
  {
    id: 'perceptron',
    name: 'Perceptron',
    depth: 1200,
    image: 'perceptron.webp',
    description: 'The simplest neural network unit',
    category: 'neural-networks',
    size: 'small'
  },
  {
    id: 'neural-network',
    name: 'Neural Network',
    depth: 1600,
    image: 'neural-network.webp',
    description: 'Multi-layer networks that learn representations',
    category: 'neural-networks',
    size: 'large'
  },
  {
    id: 'backpropagation',
    name: 'Backpropagation',
    depth: 2000,
    image: 'backpropagation.webp',
    description: 'Computing gradients through the chain rule',
    category: 'neural-networks',
    size: 'medium'
  },
  {
    id: 'activation-functions',
    name: 'Activation Functions',
    depth: 2400,
    image: 'activation-functions.webp',
    description: 'ReLU, Sigmoid, Tanh - introducing non-linearity',
    category: 'neural-networks',
    size: 'small'
  },
  {
    id: 'dropout',
    name: 'Dropout',
    depth: 2800,
    image: 'dropout.webp',
    description: 'Regularization by randomly dropping neurons',
    category: 'neural-networks',
    size: 'small'
  },
  
  // === MIDNIGHT ZONE (3000-6000m): Modern Architectures ===
  {
    id: 'cnn',
    name: 'Convolutional Neural Network',
    depth: 3200,
    image: 'cnn.webp',
    description: 'Spatial feature extraction for images',
    category: 'computer-vision',
    size: 'large'
  },
  {
    id: 'pooling',
    name: 'Pooling Layers',
    depth: 3600,
    image: 'pooling.webp',
    description: 'Downsampling to reduce spatial dimensions',
    category: 'computer-vision',
    size: 'small'
  },
  {
    id: 'rnn',
    name: 'Recurrent Neural Network',
    depth: 4000,
    image: 'rnn.webp',
    description: 'Processing sequential data with memory',
    category: 'sequence-models',
    size: 'medium'
  },
  {
    id: 'lstm',
    name: 'LSTM',
    depth: 4400,
    image: 'lstm.webp',
    description: 'Long Short-Term Memory for long sequences',
    category: 'sequence-models',
    size: 'medium'
  },
  {
    id: 'attention',
    name: 'Attention Mechanism',
    depth: 4800,
    image: 'attention.webp',
    description: 'Focusing on relevant parts of input',
    category: 'attention',
    size: 'large'
  },
  {
    id: 'transformer',
    name: 'Transformer',
    depth: 5200,
    image: 'transformer.webp',
    description: 'Self-attention architecture powering modern NLP',
    category: 'attention',
    size: 'large'
  },
  {
    id: 'bert',
    name: 'BERT',
    depth: 5600,
    image: 'bert.webp',
    description: 'Bidirectional encoder for language understanding',
    category: 'language-models',
    size: 'medium'
  },
  
  // === ABYSSAL ZONE (6000-9000m): Advanced & Generative ===
  {
    id: 'gpt',
    name: 'GPT',
    depth: 6200,
    image: 'gpt.webp',
    description: 'Generative Pre-trained Transformer',
    category: 'language-models',
    size: 'large'
  },
  {
    id: 'gan',
    name: 'GAN',
    depth: 6600,
    image: 'gan.webp',
    description: 'Generator vs Discriminator for synthetic data',
    category: 'generative',
    size: 'medium'
  },
  {
    id: 'vae',
    name: 'Variational Autoencoder',
    depth: 7000,
    image: 'vae.webp',
    description: 'Learning latent representations probabilistically',
    category: 'generative',
    size: 'medium'
  },
  {
    id: 'diffusion',
    name: 'Diffusion Models',
    depth: 7400,
    image: 'diffusion.webp',
    description: 'Iterative denoising for image generation',
    category: 'generative',
    size: 'large'
  },
  {
    id: 'reinforcement-learning',
    name: 'Reinforcement Learning',
    depth: 7800,
    image: 'reinforcement-learning.webp',
    description: 'Learning through rewards and penalties',
    category: 'rl',
    size: 'medium'
  },
  {
    id: 'q-learning',
    name: 'Q-Learning',
    depth: 8200,
    image: 'q-learning.webp',
    description: 'Learning action-value functions',
    category: 'rl',
    size: 'small'
  },
  
  // === HADAL ZONE (9000m+): Cutting Edge ===
  {
    id: 'rlhf',
    name: 'RLHF',
    depth: 9200,
    image: 'rlhf.webp',
    description: 'Reinforcement Learning from Human Feedback',
    category: 'alignment',
    size: 'medium'
  },
  {
    id: 'constitutional-ai',
    name: 'Constitutional AI',
    depth: 9600,
    image: 'constitutional-ai.webp',
    description: 'AI systems that follow principles and values',
    category: 'alignment',
    size: 'medium'
  },
  {
    id: 'mixture-of-experts',
    name: 'Mixture of Experts',
    depth: 10000,
    image: 'mixture-of-experts.webp',
    description: 'Routing inputs to specialized sub-models',
    category: 'architecture',
    size: 'large'
  },
  {
    id: 'sparse-autoencoders',
    name: 'Sparse Autoencoders',
    depth: 10400,
    image: 'sparse-autoencoders.webp',
    description: 'Interpreting neural network internals',
    category: 'interpretability',
    size: 'small'
  }
];

// Generate depth markers every 200m
export const depthMarkers = Array.from({ length: 55 }, (_, i) => i * 200);

// Zone definitions for background gradients
export const zones = [
  { name: 'Hello World', start: 0, end: 1000, color: '#1a5f7a' },
  { name: 'Actually Good', start: 1000, end: 3000, color: '#0d2f44' },
  { name: 'Understanding Code', start: 3000, end: 6000, color: '#0a1929' },
  { name: 'Abyssal', start: 6000, end: 9000, color: '#050a14' },
  { name: 'Touching Grass', start: 9000, end: 11000, color: '#000000' }
];


// Phase 3: Visual Design Ideas

// Color Scheme (Ocean Depth Gradient):
// - Surface (0-1000m): #1a5f7a → #0d4f6b (Blue-teal)
// - Twilight (1000-3000m): #0d2f44 → #0a2439 (Deep blue)
// - Midnight (3000-6000m): #0a1929 → #070f1a (Navy black)
// - Abyssal (6000-9000m): #050a14 → #03050a (Nearly black)
// - Hadal (9000m+): #000000 (Pure black)

// **Concept Sizes:**
// - large: 120-150px (major concepts like Transformers, CNNs)
// - medium: 80-100px (supporting concepts)
// - small: 50-60px (technical details)

// **Image Style Suggestions:**
// - Use **line art** or **minimalist icons** in white/cyan
// - Glowing effect for deeper concepts
// - Semi-transparent to blend with ocean
// - Consider adding subtle animations (float, pulse)

// ### **Phase 4: Image Preparation Checklist**

// **Before Development:**
// 1. Create/find 30-40 ML concept icons
// 2. Convert all to **WebP format** (90% quality)
// 3. Target size: **30-80KB per image**
// 4. Dimensions: 
//    - Large: 150x150px
//    - Medium: 100x100px
//    - Small: 60x60px
// 5. Use tools: Squoosh.app or ImageMagick

// gradient-descent.webp
// neural-network.webp
// backpropagation.webp
// transformer.webp