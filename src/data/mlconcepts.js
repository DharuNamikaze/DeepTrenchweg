// src/data/mlConcepts.js

export const mlConcepts = [
  // === SURFACE ZONE (0-1000m): Fundamentals & Classical ML ===
  {
    id: 'linear-regression',
    name: 'Linear Regression',
    depth: 100,
    image: 'linear.webp',
    description: 'Finding the best-fit line through data points',
    category: 'classical-ml',
    size: 'large'
  },
  {
    id: 'logistic-regression',
    name: 'Logistic Regression',
    depth: 180,
    image: 'logistic.webp',
    description: 'Classification using the sigmoid function',
    category: 'classical-ml',
    size: 'large'
  },
  {
    id: 'decision-trees',
    name: 'Decision Trees',
    depth: 260,
    image: 'decision.webp',
    description: 'Tree-based decisions for classification',
    category: 'classical-ml',
    size: 'large'
  },
  {
    id: 'random-forests',
    name: 'Random Forests',
    depth: 340,
    image: 'random.webp',
    description: 'Ensemble of decision trees for robust predictions',
    category: 'ensemble',
    size: 'large'
  },
  {
    id: 'svms',
    name: 'SVMs',
    depth: 420,
    image: 'svm.webp',
    description: 'Support Vector Machines for optimal hyperplanes',
    category: 'classical-ml',
    size: 'large'
  },
  {
    id: 'knn',
    name: 'KNN',
    depth: 500,
    image: 'knn.webp',
    description: 'K-Nearest Neighbors for instance-based learning',
    category: 'classical-ml',
    size: 'large'
  },
  {
    id: 'naive-bayes',
    name: 'Naive Bayes',
    depth: 580,
    image: 'naiveBayes.webp',
    description: 'Probabilistic classifier based on Bayes theorem',
    category: 'classical-ml',
    size: 'large'
  },
  {
    id: 'pca',
    name: 'PCA',
    depth: 660,
    image: 'pca.webp',
    description: 'Principal Component Analysis for dimensionality reduction',
    category: 'dimensionality-reduction',
    size: 'medium'
  },
  {
    id: 'ica',
    name: 'ICA',
    depth: 740,
    image: 'ica.webp',
    description: 'Independent Component Analysis for signal separation',
    category: 'dimensionality-reduction',
    size: 'small'
  },
  {
    id: 't-sne',
    name: 't-SNE',
    depth: 820,
    image: 't-sne.webp',
    description: 'Visualization technique for high-dimensional data',
    category: 'dimensionality-reduction',
    size: 'medium'
  },
  {
    id: 'umap',
    name: 'UMAP',
    depth: 900,
    image: 'umap.webp',
    description: 'Uniform Manifold Approximation for better embeddings',
    category: 'dimensionality-reduction',
    size: 'medium'
  },

  // === TWILIGHT ZONE (1000-3000m): Advanced ML & Early Deep Learning ===
  {
    id: 'gradient-boosting',
    name: 'Gradient Boosting',
    depth: 1050,
    image: 'gradient-boosting.webp',
    description: 'Sequential ensemble learning technique',
    category: 'ensemble',
    size: 'large'
  },
  {
    id: 'xgboost',
    name: 'XGBoost',
    depth: 1150,
    image: 'xgboost.webp',
    description: 'Extreme Gradient Boosting for structured data',
    category: 'ensemble',
    size: 'large'
  },
  {
    id: 'lightgbm',
    name: 'LightGBM',
    depth: 1250,
    image: 'lightgbm.webp',
    description: 'Light Gradient Boosting Machine for speed',
    category: 'ensemble',
    size: 'medium'
  },
  {
    id: 'catboost',
    name: 'CatBoost',
    depth: 1350,
    image: 'catboost.webp',
    description: 'Categorical Boosting for categorical features',
    category: 'ensemble',
    size: 'medium'
  },
  {
    id: 'cnns',
    name: 'CNNs',
    depth: 1450,
    image: 'cnns.webp',
    description: 'Convolutional Neural Networks for spatial data',
    category: 'computer-vision',
    size: 'large'
  },
  {
    id: 'rnns',
    name: 'RNNs',
    depth: 1550,
    image: 'rnns.webp',
    description: 'Recurrent Neural Networks for sequences',
    category: 'sequence-models',
    size: 'large'
  },
  {
    id: 'lstms',
    name: 'LSTMs',
    depth: 1650,
    image: 'lstms.webp',
    description: 'Long Short-Term Memory for long sequences',
    category: 'sequence-models',
    size: 'medium'
  },
  {
    id: 'grus',
    name: 'GRUs',
    depth: 1750,
    image: 'grus.webp',
    description: 'Gated Recurrent Units for efficient sequences',
    category: 'sequence-models',
    size: 'medium'
  },
  {
    id: 'transformers',
    name: 'Transformers',
    depth: 1850,
    image: 'transformers.webp',
    description: 'Self-attention architecture powering modern AI',
    category: 'attention',
    size: 'large'
  },
  {
    id: 'self-attention',
    name: 'Self-Attention',
    depth: 1950,
    image: 'self-attention.webp',
    description: 'Mechanism for relating different positions',
    category: 'attention',
    size: 'medium'
  },
  {
    id: 'vit',
    name: 'ViT',
    depth: 2050,
    image: 'vit.webp',
    description: 'Vision Transformer for image understanding',
    category: 'computer-vision',
    size: 'large'
  },
  {
    id: 'convnext',
    name: 'ConvNeXt',
    depth: 2150,
    image: 'convnext.webp',
    description: 'Modern ConvNet matching Transformers',
    category: 'computer-vision',
    size: 'medium'
  },
  {
    id: 'resnet',
    name: 'ResNet',
    depth: 2250,
    image: 'resnet.webp',
    description: 'Residual Networks with skip connections',
    category: 'computer-vision',
    size: 'medium'
  },
  {
    id: 'densenet',
    name: 'DenseNet',
    depth: 2350,
    image: 'densenet.webp',
    description: 'Dense connections between layers',
    category: 'computer-vision',
    size: 'small'
  },
  {
    id: 'mobilenet',
    name: 'MobileNet',
    depth: 2450,
    image: 'mobilenet.webp',
    description: 'Efficient networks for mobile devices',
    category: 'computer-vision',
    size: 'small'
  },
  {
    id: 'nas',
    name: 'NAS',
    depth: 2550,
    image: 'nas.webp',
    description: 'Neural Architecture Search for optimal designs',
    category: 'automl',
    size: 'medium'
  },
  {
    id: 'automl',
    name: 'AutoML',
    depth: 2650,
    image: 'automl.webp',
    description: 'Automated Machine Learning pipelines',
    category: 'automl',
    size: 'medium'
  },
  {
    id: 'meta-learning',
    name: 'Meta-Learning',
    depth: 2750,
    image: 'meta-learning.webp',
    description: 'Learning to learn from few examples',
    category: 'meta-learning',
    size: 'medium'
  },
  {
    id: 'few-shot-learning',
    name: 'Few-Shot Learning',
    depth: 2850,
    image: 'few-shot-learning.webp',
    description: 'Learning from minimal examples',
    category: 'meta-learning',
    size: 'small'
  },
  {
    id: 'zero-shot-learning',
    name: 'Zero-Shot Learning',
    depth: 2950,
    image: 'zero-shot-learning.webp',
    description: 'Generalizing to unseen classes',
    category: 'meta-learning',
    size: 'small'
  },

  // === MIDNIGHT ZONE (3000-6000m): Modern Deep Learning ===
  {
    id: 'contrastive-learning',
    name: 'Contrastive Learning',
    depth: 3100,
    image: 'contrastive-learning.webp',
    description: 'Learning by comparing similar and dissimilar pairs',
    category: 'representation-learning',
    size: 'large'
  },
  {
    id: 'metric-learning',
    name: 'Metric Learning',
    depth: 3220,
    image: 'metric-learning.webp',
    description: 'Learning distance metrics for embeddings',
    category: 'representation-learning',
    size: 'medium'
  },
  {
    id: 'representation-learning',
    name: 'Representation Learning',
    depth: 3340,
    image: 'representation-learning.webp',
    description: 'Learning meaningful feature representations',
    category: 'representation-learning',
    size: 'medium'
  },
  {
    id: 'foundation-models',
    name: 'Foundation Models',
    depth: 3460,
    image: 'foundation-models.webp',
    description: 'Large-scale models for multiple tasks',
    category: 'foundation-models',
    size: 'large'
  },
  {
    id: 'diffusion-models',
    name: 'Diffusion Models',
    depth: 3580,
    image: 'diffusion-models.webp',
    description: 'Iterative denoising for generation',
    category: 'generative',
    size: 'large'
  },
  {
    id: 'latent-diffusion',
    name: 'Latent Diffusion',
    depth: 3700,
    image: 'latent-diffusion.webp',
    description: 'Diffusion in compressed latent space',
    category: 'generative',
    size: 'medium'
  },
  {
    id: 'normalizing-flows',
    name: 'Normalizing Flows',
    depth: 3820,
    image: 'normalizing-flows.webp',
    description: 'Invertible transformations for generation',
    category: 'generative',
    size: 'medium'
  },
  {
    id: 'gans',
    name: 'GANs',
    depth: 3940,
    image: 'gans.webp',
    description: 'Generative Adversarial Networks',
    category: 'generative',
    size: 'large'
  },
  {
    id: 'vae',
    name: 'VAE',
    depth: 4060,
    image: 'vae.webp',
    description: 'Variational Autoencoders for latent learning',
    category: 'generative',
    size: 'medium'
  },
  {
    id: 'score-matching',
    name: 'Score Matching',
    depth: 4180,
    image: 'score-matching.webp',
    description: 'Learning data distribution gradients',
    category: 'generative',
    size: 'small'
  },
  {
    id: 'neural-odes',
    name: 'Neural ODEs',
    depth: 4300,
    image: 'neural-odes.webp',
    description: 'Continuous-depth neural networks',
    category: 'advanced-architectures',
    size: 'medium'
  },
  {
    id: 'neural-sdes',
    name: 'Neural SDEs',
    depth: 4420,
    image: 'neural-sdes.webp',
    description: 'Stochastic Differential Equations in neural nets',
    category: 'advanced-architectures',
    size: 'small'
  },
  {
    id: 'pinns',
    name: 'PINNs',
    depth: 4540,
    image: 'pinns.webp',
    description: 'Physics-Informed Neural Networks',
    category: 'scientific-ml',
    size: 'medium'
  },
  {
    id: 'kolmogorov-arnold-networks',
    name: 'Kolmogorov–Arnold Networks',
    depth: 4660,
    image: 'kolmogorov-arnold-networks.webp',
    description: 'KANs: learnable activation functions on edges',
    category: 'advanced-architectures',
    size: 'medium'
  },
  {
    id: 'mixture-of-experts',
    name: 'Mixture-of-Experts',
    depth: 4780,
    image: 'mixture-of-experts.webp',
    description: 'Routing inputs to specialized sub-models',
    category: 'advanced-architectures',
    size: 'large'
  },
  {
    id: 'routing-networks',
    name: 'Routing Networks',
    depth: 4900,
    image: 'routing-networks.webp',
    description: 'Dynamic routing based on input',
    category: 'advanced-architectures',
    size: 'small'
  },
  {
    id: 'reinforcement-learning',
    name: 'Reinforcement Learning',
    depth: 5020,
    image: 'reinforcement-learning.webp',
    description: 'Learning through rewards and penalties',
    category: 'rl',
    size: 'large'
  },
  {
    id: 'policy-gradient',
    name: 'Policy Gradient',
    depth: 5140,
    image: 'policy-gradient.webp',
    description: 'Directly optimizing policy parameters',
    category: 'rl',
    size: 'medium'
  },
  {
    id: 'actor-critic',
    name: 'Actor-Critic',
    depth: 5260,
    image: 'actor-critic.webp',
    description: 'Combining value and policy methods',
    category: 'rl',
    size: 'medium'
  },
  {
    id: 'td-learning',
    name: 'TD-Learning',
    depth: 5380,
    image: 'td-learning.webp',
    description: 'Temporal Difference learning methods',
    category: 'rl',
    size: 'small'
  },
  {
    id: 'q-learning',
    name: 'Q-Learning',
    depth: 5500,
    image: 'q-learning.webp',
    description: 'Learning action-value functions',
    category: 'rl',
    size: 'medium'
  },
  {
    id: 'deep-q-networks',
    name: 'Deep Q Networks',
    depth: 5620,
    image: 'deep-q-networks.webp',
    description: 'Q-Learning with deep neural networks',
    category: 'rl',
    size: 'medium'
  },
  {
    id: 'imitation-learning',
    name: 'Imitation Learning',
    depth: 5740,
    image: 'imitation-learning.webp',
    description: 'Learning from expert demonstrations',
    category: 'rl',
    size: 'small'
  },
  {
    id: 'inverse-rl',
    name: 'Inverse RL',
    depth: 5860,
    image: 'inverse-rl.webp',
    description: 'Learning reward functions from behavior',
    category: 'rl',
    size: 'small'
  },
  {
    id: 'world-models',
    name: 'World Models',
    depth: 5980,
    image: 'world-models.webp',
    description: 'Learning environment dynamics',
    category: 'rl',
    size: 'medium'
  },

  // === ABYSSAL ZONE (6000-9000m): Specialized & Advanced Topics ===
  {
    id: 'graph-neural-networks',
    name: 'Graph Neural Networks',
    depth: 6120,
    image: 'graph-neural-networks.webp',
    description: 'Neural networks for graph-structured data',
    category: 'graph-ml',
    size: 'large'
  },
  {
    id: 'message-passing',
    name: 'Message Passing',
    depth: 6240,
    image: 'message-passing.webp',
    description: 'Information exchange in graph networks',
    category: 'graph-ml',
    size: 'medium'
  },
  {
    id: 'graph-transformers',
    name: 'Graph Transformers',
    depth: 6360,
    image: 'graph-transformers.webp',
    description: 'Attention mechanisms for graphs',
    category: 'graph-ml',
    size: 'medium'
  },
  {
    id: 'topological-ml',
    name: 'Topological ML',
    depth: 6480,
    image: 'topological-ml.webp',
    description: 'Using topology in machine learning',
    category: 'topology',
    size: 'medium'
  },
  {
    id: 'persistent-homology',
    name: 'Persistent Homology',
    depth: 6600,
    image: 'persistent-homology.webp',
    description: 'Topological data analysis technique',
    category: 'topology',
    size: 'small'
  },
  {
    id: 'simplicial-complex-networks',
    name: 'Simplicial Complex Networks',
    depth: 6720,
    image: 'simplicial-complex-networks.webp',
    description: 'Higher-order network structures',
    category: 'topology',
    size: 'small'
  },
  {
    id: 'causal-ml',
    name: 'Causal ML',
    depth: 6840,
    image: 'causal-ml.webp',
    description: 'Learning causal relationships',
    category: 'causality',
    size: 'large'
  },
  {
    id: 'do-calculus',
    name: 'Do-Calculus',
    depth: 6960,
    image: 'do-calculus.webp',
    description: 'Mathematical framework for causality',
    category: 'causality',
    size: 'small'
  },
  {
    id: 'counterfactuals',
    name: 'Counterfactuals',
    depth: 7080,
    image: 'counterfactuals.webp',
    description: 'What-if analysis in causal models',
    category: 'causality',
    size: 'small'
  },
  {
    id: 'temporal-models',
    name: 'Temporal Models',
    depth: 7200,
    image: 'temporal-models.webp',
    description: 'Models for time-dependent data',
    category: 'time-series',
    size: 'medium'
  },
  {
    id: 'time-series-transformers',
    name: 'Time-Series Transformers',
    depth: 7320,
    image: 'time-series-transformers.webp',
    description: 'Transformers for temporal sequences',
    category: 'time-series',
    size: 'medium'
  },
  {
    id: 'wavenet',
    name: 'WaveNet',
    depth: 7440,
    image: 'wavenet.webp',
    description: 'Deep generative model for audio',
    category: 'audio',
    size: 'medium'
  },
  {
    id: 'audio-models',
    name: 'Audio Models',
    depth: 7560,
    image: 'audio-models.webp',
    description: 'Neural networks for audio processing',
    category: 'audio',
    size: 'medium'
  },
  {
    id: 'speech-models',
    name: 'Speech Models',
    depth: 7680,
    image: 'speech-models.webp',
    description: 'Models for speech recognition and synthesis',
    category: 'audio',
    size: 'medium'
  },
  {
    id: 'diffusion-audio',
    name: 'Diffusion Audio',
    depth: 7800,
    image: 'diffusion-audio.webp',
    description: 'Diffusion models for audio generation',
    category: 'audio',
    size: 'small'
  },
  {
    id: 'rlhf',
    name: 'RLHF',
    depth: 7920,
    image: 'rlhf.webp',
    description: 'Reinforcement Learning from Human Feedback',
    category: 'alignment',
    size: 'large'
  },
  {
    id: 'direct-preference-optimization',
    name: 'Direct Preference Optimization',
    depth: 8040,
    image: 'direct-preference-optimization.webp',
    description: 'DPO: simpler alternative to RLHF',
    category: 'alignment',
    size: 'medium'
  },
  {
    id: 'self-play',
    name: 'Self-Play',
    depth: 8160,
    image: 'self-play.webp',
    description: 'Training agents against themselves',
    category: 'rl',
    size: 'small'
  },
  {
    id: 'curriculum-learning',
    name: 'Curriculum Learning',
    depth: 8280,
    image: 'curriculum-learning.webp',
    description: 'Learning from easy to hard examples',
    category: 'training',
    size: 'small'
  },
  {
    id: 'neural-compression',
    name: 'Neural Compression',
    depth: 8400,
    image: 'neural-compression.webp',
    description: 'Compressing data with neural networks',
    category: 'compression',
    size: 'medium'
  },
  {
    id: 'tokenization',
    name: 'Tokenization',
    depth: 8520,
    image: 'tokenization.webp',
    description: 'Breaking text into processable units',
    category: 'nlp',
    size: 'small'
  },
  {
    id: 'byte-level-models',
    name: 'Byte-Level Models',
    depth: 8640,
    image: 'byte-level-models.webp',
    description: 'Processing raw bytes instead of tokens',
    category: 'nlp',
    size: 'small'
  },
  {
    id: 'sparse-transformers',
    name: 'Sparse Transformers',
    depth: 8760,
    image: 'sparse-transformers.webp',
    description: 'Efficient attention with sparsity',
    category: 'efficiency',
    size: 'medium'
  },
  {
    id: 'flash-attention',
    name: 'Flash Attention',
    depth: 8880,
    image: 'flash-attention.webp',
    description: 'Fast and memory-efficient attention',
    category: 'efficiency',
    size: 'medium'
  },

  // === HADAL ZONE (9000m+): Cutting Edge & Research ===
  {
    id: 'kernel-methods',
    name: 'Kernel Methods',
    depth: 9020,
    image: 'kernel-methods.webp',
    description: 'Implicit feature mappings',
    category: 'theory',
    size: 'small'
  },
  {
    id: 'neural-tangent-kernel',
    name: 'Neural Tangent Kernel',
    depth: 9140,
    image: 'neural-tangent-kernel.webp',
    description: 'Understanding infinite-width networks',
    category: 'theory',
    size: 'small'
  },
  {
    id: 'implicit-layers',
    name: 'Implicit Layers',
    depth: 9260,
    image: 'implicit-layers.webp',
    description: 'Layers defined by fixed-point equations',
    category: 'advanced-architectures',
    size: 'small'
  },
  {
    id: 'deep-equilibrium-models',
    name: 'Deep Equilibrium Models',
    depth: 9380,
    image: 'deep-equilibrium-models.webp',
    description: 'Fixed-point models for deep learning',
    category: 'advanced-architectures',
    size: 'medium'
  },
  {
    id: 'large-scale-training',
    name: 'Large-Scale Training',
    depth: 9500,
    image: 'large-scale-training.webp',
    description: 'Training massive models efficiently',
    category: 'systems',
    size: 'large'
  },
  {
    id: 'sharded-training',
    name: 'Sharded Training',
    depth: 9620,
    image: 'sharded-training.webp',
    description: 'Distributing model across devices',
    category: 'systems',
    size: 'medium'
  },
  {
    id: 'fsdp',
    name: 'FSDP',
    depth: 9740,
    image: 'fsdp.webp',
    description: 'Fully Sharded Data Parallel training',
    category: 'systems',
    size: 'small'
  },
  {
    id: 'zero-offload',
    name: 'ZeRO-Offload',
    depth: 9860,
    image: 'zero-offload.webp',
    description: 'Offloading optimizer states to CPU',
    category: 'systems',
    size: 'small'
  },
  {
    id: 'quantization',
    name: 'Quantization',
    depth: 9980,
    image: 'quantization.webp',
    description: 'Reducing precision for efficiency',
    category: 'optimization',
    size: 'medium'
  },
  {
    id: 'pruning',
    name: 'Pruning',
    depth: 10100,
    image: 'pruning.webp',
    description: 'Removing unnecessary connections',
    category: 'optimization',
    size: 'small'
  },
  {
    id: 'distillation',
    name: 'Distillation',
    depth: 10220,
    image: 'distillation.webp',
    description: 'Transferring knowledge to smaller models',
    category: 'optimization',
    size: 'medium'
  },
  {
    id: 'low-rank-adaptation',
    name: 'Low-Rank Adaptation',
    depth: 10340,
    image: 'low-rank-adaptation.webp',
    description: 'LoRA: efficient fine-tuning method',
    category: 'fine-tuning',
    size: 'medium'
  },
  {
    id: 'adapters',
    name: 'Adapters',
    depth: 10460,
    image: 'adapters.webp',
    description: 'Lightweight modules for task adaptation',
    category: 'fine-tuning',
    size: 'small'
  },
  {
    id: 'prompting',
    name: 'Prompting',
    depth: 10580,
    image: 'prompting.webp',
    description: 'Guiding models with natural language',
    category: 'fine-tuning',
    size: 'small'
  },
  {
    id: 'fine-tuning',
    name: 'Fine-Tuning',
    depth: 10700,
    image: 'fine-tuning.webp',
    description: 'Adapting pre-trained models to tasks',
    category: 'fine-tuning',
    size: 'medium'
  },
  {
    id: 'continual-learning',
    name: 'Continual Learning',
    depth: 10820,
    image: 'continual-learning.webp',
    description: 'Learning sequentially without forgetting',
    category: 'lifelong-learning',
    size: 'medium'
  },
  {
    id: 'catastrophic-forgetting',
    name: 'Catastrophic Forgetting',
    depth: 10940,
    image: 'catastrophic-forgetting.webp',
    description: 'Challenge of retaining old knowledge',
    category: 'lifelong-learning',
    size: 'small'
  },
  {
    id: 'lifelong-rl',
    name: 'Lifelong RL',
    depth: 11060,
    image: 'lifelong-rl.webp',
    description: 'Reinforcement learning across tasks',
    category: 'lifelong-learning',
    size: 'small'
  },
  {
    id: 'federated-learning',
    name: 'Federated Learning',
    depth: 11180,
    image: 'federated-learning.webp',
    description: 'Decentralized model training',
    category: 'distributed-ml',
    size: 'medium'
  },
  {
    id: 'edge-ml',
    name: 'Edge ML',
    depth: 11300,
    image: 'edge-ml.webp',
    description: 'Machine learning on edge devices',
    category: 'distributed-ml',
    size: 'small'
  },
  {
    id: 'tinyml',
    name: 'TinyML',
    depth: 11420,
    image: 'tinyml.webp',
    description: 'ML on microcontrollers',
    category: 'distributed-ml',
    size: 'small'
  },
  {
    id: 'on-device-diffusion',
    name: 'On-Device Diffusion',
    depth: 11540,
    image: 'on-device-diffusion.webp',
    description: 'Running diffusion models locally',
    category: 'distributed-ml',
    size: 'small'
  },
  {
    id: 'secure-ml',
    name: 'Secure ML',
    depth: 11660,
    image: 'secure-ml.webp',
    description: 'Privacy-preserving machine learning',
    category: 'security',
    size: 'medium'
  },
  {
    id: 'homomorphic-encryption',
    name: 'Homomorphic Encryption',
    depth: 11780,
    image: 'homomorphic-encryption.webp',
    description: 'Computing on encrypted data',
    category: 'security',
    size: 'small'
  },
  {
    id: 'differential-privacy',
    name: 'Differential Privacy',
    depth: 11900,
    image: 'differential-privacy.webp',
    description: 'Statistical privacy guarantees',
    category: 'security',
    size: 'small'
  },
  {
    id: 'ai-compilers',
    name: 'AI Compilers',
    depth: 12020,
    image: 'ai-compilers.webp',
    description: 'Optimizing ML model execution',
    category: 'compilers',
    size: 'medium'
  },
  {
    id: 'xla',
    name: 'XLA',
    depth: 12140,
    image: 'xla.webp',
    description: 'Accelerated Linear Algebra compiler',
    category: 'compilers',
    size: 'small'
  },
  {
    id: 'tvm',
    name: 'TVM',
    depth: 12260,
    image: 'tvm.webp',
    description: 'End-to-end deep learning compiler',
    category: 'compilers',
    size: 'small'
  },
  {
    id: 'tensorrt',
    name: 'TensorRT',
    depth: 12380,
    image: 'tensorrt.webp',
    description: 'NVIDIA inference optimization SDK',
    category: 'compilers',
    size: 'small'
  },
  {
    id: 'mlir',
    name: 'MLIR',
    depth: 12500,
    image: 'mlir.webp',
    description: 'Multi-Level Intermediate Representation',
    category: 'compilers',
    size: 'small'
  },
  {
    id: 'onnx-runtime',
    name: 'ONNX Runtime',
    depth: 12620,
    image: 'onnx-runtime.webp',
    description: 'Cross-platform inference engine',
    category: 'compilers',
    size: 'small'
  },
  {
    id: 'glow-compiler',
    name: 'Glow Compiler',
    depth: 12740,
    image: 'glow-compiler.webp',
    description: 'Machine learning compiler for hardware',
    category: 'compilers',
    size: 'small'
  },
  {
    id: 'graph-optimization',
    name: 'Graph Optimization',
    depth: 12860,
    image: 'graph-optimization.webp',
    description: 'Optimizing computational graphs',
    category: 'optimization',
    size: 'small'
  },
  {
    id: 'operator-fusion',
    name: 'Operator Fusion',
    depth: 12980,
    image: 'operator-fusion.webp',
    description: 'Combining operations for efficiency',
    category: 'optimization',
    size: 'small'
  },
  {
    id: 'kernel-fusion',
    name: 'Kernel Fusion',
    depth: 13100,
    image: 'kernel-fusion.webp',
    description: 'Merging GPU kernels',
    category: 'optimization',
    size: 'small'
  },
  {
    id: 'memory-planning',
    name: 'Memory Planning',
    depth: 13220,
    image: 'memory-planning.webp',
    description: 'Optimizing memory allocation',
    category: 'optimization',
    size: 'small'
  },
  {
    id: 'scheduling',
    name: 'Scheduling',
    depth: 13340,
    image: 'scheduling.webp',
    description: 'Optimal execution order planning',
    category: 'optimization',
    size: 'small'
  },
  {
    id: 'high-performance-inference',
    name: 'High-Performance Inference',
    depth: 13460,
    image: 'high-performance-inference.webp',
    description: 'Ultra-fast model serving',
    category: 'inference',
    size: 'medium'
  },
  {
    id: 'sparse-kernels',
    name: 'Sparse Kernels',
    depth: 13580,
    image: 'sparse-kernels.webp',
    description: 'Efficient operations on sparse data',
    category: 'inference',
    size: 'small'
  },
  {
    id: 'flash-decoding',
    name: 'Flash-Decoding',
    depth: 13700,
    image: 'flash-decoding.webp',
    description: 'Accelerated sequence generation',
    category: 'inference',
    size: 'small'
  },
  {
    id: 'speculative-decoding',
    name: 'Speculative Decoding',
    depth: 13820,
    image: 'speculative-decoding.webp',
    description: 'Parallel generation with verification',
    category: 'inference',
    size: 'small'
  },
  {
    id: 'caching-transformers',
    name: 'Caching Transformers',
    depth: 13940,
    image: 'caching-transformers.webp',
    description: 'KV-cache optimization',
    category: 'inference',
    size: 'small'
  },
  {
    id: 'neural-rendering',
    name: 'Neural Rendering',
    depth: 14060,
    image: 'neural-rendering.webp',
    description: 'Synthesizing images with neural networks',
    category: '3d-vision',
    size: 'medium'
  },
  {
    id: 'nerf',
    name: 'NeRF',
    depth: 14180,
    image: 'nerf.webp',
    description: 'Neural Radiance Fields for 3D scenes',
    category: '3d-vision',
    size: 'large'
  },
  {
    id: '3d-diffusion',
    name: '3D Diffusion',
    depth: 14300,
    image: '3d-diffusion.webp',
    description: 'Diffusion models for 3D generation',
    category: '3d-vision',
    size: 'medium'
  },
  {
    id: 'gaussian-splatting',
    name: 'Gaussian Splatting',
    depth: 14420,
    image: 'gaussian-splatting.webp',
    description: 'Real-time 3D reconstruction',
    category: '3d-vision',
    size: 'medium'
  },
  {
    id: 'image-conditioned-models',
    name: 'Image Conditioned Models',
    depth: 14540,
    image: 'image-conditioned-models.webp',
    description: 'Models guided by reference images',
    category: 'multimodal',
    size: 'small'
  },
  {
    id: 'multimodal-models',
    name: 'Multimodal Models',
    depth: 14660,
    image: 'multimodal-models.webp',
    description: 'Processing multiple data types',
    category: 'multimodal',
    size: 'large'
  },
  {
    id: 'clip',
    name: 'CLIP',
    depth: 14780,
    image: 'clip.webp',
    description: 'Contrastive Language-Image Pre-training',
    category: 'multimodal',
    size: 'large'
  },
  {
    id: 'image-text-models',
    name: 'Image+Text Models',
    depth: 14900,
    image: 'image-text-models.webp',
    description: 'Joint vision-language understanding',
    category: 'multimodal',
    size: 'medium'
  },
  {
    id: 'video-transformers',
    name: 'Video Transformers',
    depth: 15020,
    image: 'video-transformers.webp',
    description: 'Transformers for video understanding',
    category: 'video',
    size: 'medium'
  },
  {
    id: '3d-vision-transformers',
    name: '3D Vision Transformers',
    depth: 15140,
    image: '3d-vision-transformers.webp',
    description: 'ViT for 3D data',
    category: '3d-vision',
    size: 'small'
  },
  {
    id: 'ultra-long-context-models',
    name: 'Ultra-Long Context Models',
    depth: 15260,
    image: 'ultra-long-context-models.webp',
    description: 'Processing millions of tokens',
    category: 'long-context',
    size: 'large'
  },
  {
    id: 'state-space-models',
    name: 'State-Space Models',
    depth: 15380,
    image: 'state-space-models.webp',
    description: 'Efficient sequence modeling',
    category: 'long-context',
    size: 'medium'
  },
  {
    id: 's4',
    name: 'S4',
    depth: 15500,
    image: 's4.webp',
    description: 'Structured State Space for sequences',
    category: 'long-context',
    size: 'medium'
  },
  {
    id: 'hyena',
    name: 'Hyena',
    depth: 15620,
    image: 'hyena.webp',
    description: 'Sub-quadratic attention alternative',
    category: 'long-context',
    size: 'small'
  },
  {
    id: 'rwkv',
    name: 'RWKV',
    depth: 15740,
    image: 'rwkv.webp',
    description: 'Receptance Weighted Key Value',
    category: 'long-context',
    size: 'small'
  },
  {
    id: 'recurrent-transformers',
    name: 'Recurrent Transformers',
    depth: 15860,
    image: 'recurrent-transformers.webp',
    description: 'Combining recurrence with attention',
    category: 'long-context',
    size: 'small'
  },
  {
    id: 'neural-pde-solvers',
    name: 'Neural PDE Solvers',
    depth: 15980,
    image: 'neural-pde-solvers.webp',
    description: 'Solving partial differential equations',
    category: 'scientific-ml',
    size: 'medium'
  },
  {
    id: 'physics-informed-transformers',
    name: 'Physics-Informed Transformers',
    depth: 16100,
    image: 'physics-informed-transformers.webp',
    description: 'Incorporating physics into transformers',
    category: 'scientific-ml',
    size: 'small'
  },
  {
    id: 'symbolic-regression',
    name: 'Symbolic Regression',
    depth: 16220,
    image: 'symbolic-regression.webp',
    description: 'Discovering mathematical equations',
    category: 'scientific-ml',
    size: 'medium'
  },
  {
    id: 'automated-theorem-proving',
    name: 'Automated Theorem Proving',
    depth: 16340,
    image: 'automated-theorem-proving.webp',
    description: 'AI for mathematical proofs',
    category: 'scientific-ml',
    size: 'small'
  },
  {
    id: 'ml-equation-solvers',
    name: 'ML Equation Solvers',
    depth: 16460,
    image: 'ml-equation-solvers.webp',
    description: 'Neural approaches to equation solving',
    category: 'scientific-ml',
    size: 'small'
  },
  {
    id: 'fourier-neural-operators',
    name: 'Fourier Neural Operators',
    depth: 16580,
    image: 'fourier-neural-operators.webp',
    description: 'Learning operators in Fourier space',
    category: 'scientific-ml',
    size: 'medium'
  },
  {
    id: 'spectral-methods-in-ml',
    name: 'Spectral Methods in ML',
    depth: 16700,
    image: 'spectral-methods-in-ml.webp',
    description: 'Frequency domain approaches',
    category: 'scientific-ml',
    size: 'small'
  },
  {
    id: 'wavelet-models',
    name: 'Wavelet Models',
    depth: 16820,
    image: 'wavelet-models.webp',
    description: 'Multi-scale analysis with wavelets',
    category: 'scientific-ml',
    size: 'small'
  },
  {
    id: 'latent-space-geometry',
    name: 'Latent Space Geometry',
    depth: 16940,
    image: 'latent-space-geometry.webp',
    description: 'Understanding learned representations',
    category: 'theory',
    size: 'small'
  },
  {
    id: 'manifold-learning',
    name: 'Manifold Learning',
    depth: 17060,
    image: 'manifold-learning.webp',
    description: 'Learning data manifolds',
    category: 'theory',
    size: 'small'
  },
  {
    id: 'geometric-deep-learning',
    name: 'Geometric Deep Learning',
    depth: 17180,
    image: 'geometric-deep-learning.webp',
    description: 'DL on non-Euclidean domains',
    category: 'theory',
    size: 'medium'
  },
  {
    id: 'lie-groups-in-dl',
    name: 'Lie Groups in DL',
    depth: 17300,
    image: 'lie-groups-in-dl.webp',
    description: 'Group theory in neural networks',
    category: 'theory',
    size: 'small'
  },
  {
    id: 'riemannian-optimization',
    name: 'Riemannian Optimization',
    depth: 17420,
    image: 'riemannian-optimization.webp',
    description: 'Optimization on manifolds',
    category: 'theory',
    size: 'small'
  },
  {
    id: 'neural-fields',
    name: 'Neural Fields',
    depth: 17540,
    image: 'neural-fields.webp',
    description: 'Continuous representations',
    category: '3d-vision',
    size: 'medium'
  },
  {
    id: 'implicit-geometry-models',
    name: 'Implicit Geometry Models',
    depth: 17660,
    image: 'implicit-geometry-models.webp',
    description: 'Neural implicit representations',
    category: '3d-vision',
    size: 'small'
  },
  {
    id: 'signed-distance-functions',
    name: 'Signed Distance Functions',
    depth: 17780,
    image: 'signed-distance-functions.webp',
    description: 'SDF for shape representation',
    category: '3d-vision',
    size: 'small'
  },
  {
    id: 'volumetric-rendering',
    name: 'Volumetric Rendering',
    depth: 17900,
    image: 'volumetric-rendering.webp',
    description: 'Rendering 3D volumes',
    category: '3d-vision',
    size: 'small'
  },
  {
    id: 'scientific-ml',
    name: 'Scientific ML',
    depth: 18020,
    image: 'scientific-ml.webp',
    description: 'ML for scientific discovery',
    category: 'scientific-ml',
    size: 'large'
  },
  {
    id: 'topological-priors',
    name: 'Topological Priors',
    depth: 18140,
    image: 'topological-priors.webp',
    description: 'Using topology as inductive bias',
    category: 'topology',
    size: 'small'
  },
  {
    id: 'structured-state-space-models',
    name: 'Structured State-Space Models',
    depth: 18260,
    image: 'structured-state-space-models.webp',
    description: 'Efficient recurrent architectures',
    category: 'long-context',
    size: 'medium'
  }
];

// Generate depth markers every 200m
export const depthMarkers = Array.from({ length: 92 }, (_, i) => i * 200);

// Zone definitions for background gradients
export const zones = [
  { name: 'Hello World', start: 0, end:500, color: '#1a5f7a' },
  { name: 'Actually Good', start: 1500, end: 2000, color: '#0d2f44' },
  { name: 'Understanding Code', start: 2000, end: 3000, color: '#0a1929' },
  { name: 'Abyssal', start: 3000, end: 4000, color: '#050a14' },
  { name: 'Touching Grass', start: 4000, end: 18200, color: '#000000' }
];