/**
 * NES (Natural Evolution Strategies) Reinforcement Learning Agent
 * Optimized for legal document processing and WebAssembly integration
 */

// NES-RL Configuration
const NES_CONFIG = {
  populationSize: 50,         // Population for evolution
  learningRate: 0.01,        // Learning rate for policy updates
  noiseStdDev: 0.1,          // Standard deviation for noise
  eliteRatio: 0.2,           // Top performers ratio
  maxGenerations: 1000,      // Maximum training generations
  convergenceThreshold: 1e-6, // Convergence threshold
  parallelEvaluations: 8     // Parallel fitness evaluations
};

/**
 * NES Reinforcement Learning Agent
 * Uses evolution strategies for policy optimization
 */
class NESRLAgent {
  constructor(config = {}) {
    this.config = { ...NES_CONFIG, ...config };
    this.stateSize = config.stateSize || 384;
    this.actionSize = config.actionSize || 256;
    
    // Policy network parameters (flattened)
    this.policyParams = this.initializePolicy();
    this.bestParams = [...this.policyParams];
    this.bestFitness = -Infinity;
    
    // Evolution tracking
    this.generation = 0;
    this.fitnessHistory = [];
    this.populationFitness = [];
    
    // Experience replay buffer
    this.experienceBuffer = [];
    this.maxExperienceSize = 10000;
    
    // Exploration parameters
    this.epsilon = 1.0;
    this.epsilonMin = 0.01;
    this.epsilonDecay = 0.995;
    
    console.log('🧬 NES-RL Agent initialized with', {
      stateSize: this.stateSize,
      actionSize: this.actionSize,
      populationSize: this.config.populationSize
    });
  }

  /**
   * Initialize policy network parameters
   */
  initializePolicy() {
    const paramCount = this.calculateParamCount();
    const params = new Float32Array(paramCount);
    
    // Xavier initialization
    const scale = Math.sqrt(2.0 / (this.stateSize + this.actionSize));
    for (let i = 0; i < paramCount; i++) {
      params[i] = (Math.random() * 2 - 1) * scale;
    }
    
    return params;
  }

  /**
   * Calculate total parameter count for the neural network
   */
  calculateParamCount() {
    const hiddenSize = Math.max(64, Math.min(256, this.stateSize));
    // Input → Hidden → Output
    return (this.stateSize * hiddenSize) + hiddenSize + 
           (hiddenSize * this.actionSize) + this.actionSize;
  }

  /**
   * Forward pass through the policy network
   */
  forwardPass(state, params = this.policyParams) {
    const hiddenSize = Math.max(64, Math.min(256, this.stateSize));
    const hidden = new Float32Array(hiddenSize);
    const output = new Float32Array(this.actionSize);
    
    let paramIndex = 0;
    
    // Input to hidden layer
    for (let h = 0; h < hiddenSize; h++) {
      let sum = 0;
      for (let i = 0; i < this.stateSize; i++) {
        sum += state[i] * params[paramIndex++];
      }
      sum += params[paramIndex++]; // bias
      hidden[h] = Math.tanh(sum); // activation
    }
    
    // Hidden to output layer
    for (let o = 0; o < this.actionSize; o++) {
      let sum = 0;
      for (let h = 0; h < hiddenSize; h++) {
        sum += hidden[h] * params[paramIndex++];
      }
      sum += params[paramIndex++]; // bias
      output[o] = sum; // linear output
    }
    
    return this.softmax(output);
  }

  /**
   * Softmax activation for action probabilities
   */
  softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expLogits = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expLogits.reduce((sum, x) => sum + x, 0);
    return expLogits.map(x => x / sumExp);
  }

  /**
   * Select action using current policy
   */
  selectAction(state) {
    const actionProbs = this.forwardPass(state);
    
    // Epsilon-greedy exploration
    if (Math.random() < this.epsilon) {
      // Random action for exploration
      const randomAction = Math.floor(Math.random() * this.actionSize);
      return {
        action: randomAction,
        probability: actionProbs[randomAction],
        temperature: 0.8 + Math.random() * 0.4,
        maxTokens: 128 + Math.floor(Math.random() * 128),
        explorationBonus: 0.1
      };
    }
    
    // Sample from policy distribution
    const action = this.sampleFromDistribution(actionProbs);
    
    return {
      action: action,
      probability: actionProbs[action],
      temperature: this.adaptiveTemperature(actionProbs),
      maxTokens: this.adaptiveMaxTokens(state),
      explorationBonus: 0
    };
  }

  /**
   * Sample action from probability distribution
   */
  sampleFromDistribution(probs) {
    const random = Math.random();
    let cumulative = 0;
    
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (random <= cumulative) {
        return i;
      }
    }
    
    return probs.length - 1;
  }

  /**
   * Adaptive temperature based on action distribution
   */
  adaptiveTemperature(actionProbs) {
    // Higher entropy → higher temperature (more exploration)
    const entropy = -actionProbs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
    const maxEntropy = Math.log(this.actionSize);
    const normalizedEntropy = entropy / maxEntropy;
    
    return 0.3 + normalizedEntropy * 0.7; // Range: 0.3 to 1.0
  }

  /**
   * Adaptive max tokens based on state complexity
   */
  adaptiveMaxTokens(state) {
    // Estimate complexity from state embedding
    const complexity = state.reduce((sum, x) => sum + Math.abs(x), 0) / state.length;
    const normalizedComplexity = Math.min(1, complexity * 2);
    
    return Math.floor(64 + normalizedComplexity * 192); // Range: 64 to 256
  }

  /**
   * Update policy using NES algorithm
   */
  updatePolicy(stateEmbedding, action, reward) {
    // Store experience
    this.storeExperience(stateEmbedding, action, reward);
    
    // Decay exploration
    this.epsilon = Math.max(this.epsilonMin, this.epsilon * this.epsilonDecay);
    
    // Update if we have enough experiences
    if (this.experienceBuffer.length >= this.config.populationSize) {
      this.performNESUpdate();
    }
  }

  /**
   * Store experience in replay buffer
   */
  storeExperience(state, action, reward) {
    const experience = {
      state: new Float32Array(state),
      action: action,
      reward: reward,
      timestamp: Date.now()
    };
    
    this.experienceBuffer.push(experience);
    
    // Maintain buffer size
    if (this.experienceBuffer.length > this.maxExperienceSize) {
      this.experienceBuffer.shift();
    }
  }

  /**
   * Perform NES policy update
   */
  performNESUpdate() {
    console.log(`🧬 Performing NES update - Generation ${this.generation}`);
    
    // Generate population of parameter variations
    const population = this.generatePopulation();
    
    // Evaluate fitness for each individual
    const fitnessScores = this.evaluatePopulation(population);
    
    // Update policy parameters using fitness-weighted average
    this.updateParametersFromPopulation(population, fitnessScores);
    
    // Track progress
    this.generation++;
    const avgFitness = fitnessScores.reduce((sum, f) => sum + f, 0) / fitnessScores.length;
    const maxFitness = Math.max(...fitnessScores);
    
    this.fitnessHistory.push({ avg: avgFitness, max: maxFitness });
    
    // Update best parameters if improved
    if (maxFitness > this.bestFitness) {
      this.bestFitness = maxFitness;
      const bestIndex = fitnessScores.indexOf(maxFitness);
      this.bestParams = [...population[bestIndex]];
    }
    
    console.log(`📊 Generation ${this.generation}: Avg=${avgFitness.toFixed(4)}, Max=${maxFitness.toFixed(4)}`);
  }

  /**
   * Generate population of parameter variations
   */
  generatePopulation() {
    const population = [];
    const noise = this.generateNoise();
    
    for (let i = 0; i < this.config.populationSize; i++) {
      const individual = new Float32Array(this.policyParams.length);
      for (let j = 0; j < individual.length; j++) {
        individual[j] = this.policyParams[j] + noise[i][j];
      }
      population.push(individual);
    }
    
    return population;
  }

  /**
   * Generate noise for population diversity
   */
  generateNoise() {
    const noise = [];
    for (let i = 0; i < this.config.populationSize; i++) {
      const individualNoise = new Float32Array(this.policyParams.length);
      for (let j = 0; j < individualNoise.length; j++) {
        individualNoise[j] = this.gaussianRandom() * this.config.noiseStdDev;
      }
      noise.push(individualNoise);
    }
    return noise;
  }

  /**
   * Generate Gaussian random number (Box-Muller transform)
   */
  gaussianRandom() {
    if (this.hasSpareGaussian) {
      this.hasSpareGaussian = false;
      return this.spareGaussian;
    }
    
    this.hasSpareGaussian = true;
    const u = Math.random();
    const v = Math.random();
    const mag = this.config.noiseStdDev * Math.sqrt(-2 * Math.log(u));
    this.spareGaussian = mag * Math.cos(2 * Math.PI * v);
    return mag * Math.sin(2 * Math.PI * v);
  }

  /**
   * Evaluate fitness for population
   */
  evaluatePopulation(population) {
    const fitnessScores = [];
    
    for (let i = 0; i < population.length; i++) {
      const individual = population[i];
      let totalFitness = 0;
      let evaluationCount = 0;
      
      // Sample experiences for evaluation
      const sampleSize = Math.min(10, this.experienceBuffer.length);
      const samples = this.sampleExperiences(sampleSize);
      
      for (const experience of samples) {
        const actionProbs = this.forwardPass(experience.state, individual);
        const actionValue = actionProbs[experience.action.action] || 0;
        
        // Fitness combines reward and action probability
        const fitness = experience.reward * actionValue;
        totalFitness += fitness;
        evaluationCount++;
      }
      
      fitnessScores.push(evaluationCount > 0 ? totalFitness / evaluationCount : 0);
    }
    
    return fitnessScores;
  }

  /**
   * Sample experiences for fitness evaluation
   */
  sampleExperiences(count) {
    if (this.experienceBuffer.length <= count) {
      return [...this.experienceBuffer];
    }
    
    const samples = [];
    for (let i = 0; i < count; i++) {
      const randomIndex = Math.floor(Math.random() * this.experienceBuffer.length);
      samples.push(this.experienceBuffer[randomIndex]);
    }
    return samples;
  }

  /**
   * Update parameters using fitness-weighted population
   */
  updateParametersFromPopulation(population, fitnessScores) {
    // Sort by fitness (descending)
    const sortedIndices = fitnessScores
      .map((fitness, index) => ({ fitness, index }))
      .sort((a, b) => b.fitness - a.fitness);
    
    // Select elite individuals
    const eliteCount = Math.floor(this.config.populationSize * this.config.eliteRatio);
    const eliteIndices = sortedIndices.slice(0, eliteCount).map(item => item.index);
    
    // Weighted average of elite parameters
    const newParams = new Float32Array(this.policyParams.length);
    let totalWeight = 0;
    
    for (let i = 0; i < eliteCount; i++) {
      const index = eliteIndices[i];
      const weight = fitnessScores[index];
      totalWeight += weight;
      
      for (let j = 0; j < newParams.length; j++) {
        newParams[j] += population[index][j] * weight;
      }
    }
    
    // Normalize and update
    if (totalWeight > 0) {
      for (let j = 0; j < newParams.length; j++) {
        this.policyParams[j] = this.config.learningRate * (newParams[j] / totalWeight) + 
                               (1 - this.config.learningRate) * this.policyParams[j];
      }
    }
  }

  /**
   * Train agent on batch of episodes
   */
  async trainBatch(episodes) {
    console.log(`🎯 Training NES agent on ${episodes.length} episodes...`);
    
    for (const episode of episodes) {
      for (const step of episode.steps) {
        await this.updatePolicy(step.state, step.action, step.reward);
      }
    }
    
    // Performance statistics
    const recentFitness = this.fitnessHistory.slice(-10);
    const avgRecentFitness = recentFitness.length > 0 
      ? recentFitness.reduce((sum, f) => sum + f.avg, 0) / recentFitness.length 
      : 0;
    
    return {
      generation: this.generation,
      avgFitness: avgRecentFitness,
      bestFitness: this.bestFitness,
      epsilon: this.epsilon,
      experienceCount: this.experienceBuffer.length
    };
  }

  /**
   * Get agent statistics
   */
  getStats() {
    return {
      generation: this.generation,
      bestFitness: this.bestFitness,
      epsilon: this.epsilon,
      experienceCount: this.experienceBuffer.length,
      parameterCount: this.policyParams.length,
      fitnessHistory: this.fitnessHistory.slice(-20), // Last 20 generations
      populationSize: this.config.populationSize,
      learningRate: this.config.learningRate
    };
  }

  /**
   * Save agent state for persistence
   */
  serialize() {
    return {
      policyParams: Array.from(this.policyParams),
      bestParams: Array.from(this.bestParams),
      bestFitness: this.bestFitness,
      generation: this.generation,
      epsilon: this.epsilon,
      fitnessHistory: this.fitnessHistory,
      config: this.config
    };
  }

  /**
   * Load agent state from serialized data
   */
  deserialize(data) {
    this.policyParams = new Float32Array(data.policyParams);
    this.bestParams = new Float32Array(data.bestParams);
    this.bestFitness = data.bestFitness;
    this.generation = data.generation;
    this.epsilon = data.epsilon;
    this.fitnessHistory = data.fitnessHistory || [];
    this.config = { ...this.config, ...data.config };
    
    console.log(`📥 NES-RL agent loaded - Generation ${this.generation}, Best Fitness: ${this.bestFitness.toFixed(4)}`);
  }
}

// Export for use in service workers
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { NESRLAgent, NES_CONFIG };
} else if (typeof self !== 'undefined') {
  self.NESRLAgent = NESRLAgent;
  self.NES_CONFIG = NES_CONFIG;
}

console.log('🧬 NES-RL module loaded successfully');