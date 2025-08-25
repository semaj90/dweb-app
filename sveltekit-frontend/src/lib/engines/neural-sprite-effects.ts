// import * as fabric from "fabric"; // Dynamic import to avoid canvas.node issues
let fabric: any;

/**
 * Neural Sprite Effects Library
 * Advanced visual effects for the NES-inspired sprite engine
 */


export interface SpriteEffect {
  id: string;
  name: string;
  description: string;
  apply: (canvas: any, options: any) => Promise<void>;
  cleanup?: (canvas: any) => void;
}

export class NeuralSpriteEffects {
  private canvas: any;
  private activeEffects: Map<string, any> = new Map();

  constructor(canvas: any) {
    this.canvas = canvas;
  }

  // Retro CRT scanlines effect
  async applyCRTScanlines(intensity: number = 0.3): Promise<void> {
    const canvasWidth = this.canvas.getWidth();
    const canvasHeight = this.canvas.getHeight();

    // Create scanlines
    for (let y = 0; y < canvasHeight; y += 4) {
      const scanline = new (fabric as any).Rect({
        left: 0,
        top: y,
        width: canvasWidth,
        height: 2,
        fill: `rgba(0, 0, 0, ${intensity})`,
        selectable: false,
        evented: false,
        excludeFromExport: true,
      });

      this.canvas.add(scanline);
    }

    this.canvas.renderAll();
  }

  // NES-style pixelation effect
  async applyPixelation(pixelSize: number = 8): Promise<void> {
    const filter = new (fabric as any).Image.filters.Pixelate({
      blocksize: pixelSize,
    });

    // Apply to all image objects
    this.canvas.getObjects().forEach((obj: any) => {
      if (obj.type === 'image') {
        obj.filters = [filter];
        obj.applyFilters();
      }
    });

    this.canvas.renderAll();
  }

  // Matrix digital rain effect (inspired by legal AI theme)
  async applyMatrixRain(
    speed: number = 50,
    color: string = "#00ff41",
  ): Promise<void> {
    const canvasWidth = this.canvas.getWidth();
    const canvasHeight = this.canvas.getHeight();
    const characters = "01ABCDEFGHIJKLMNOPQRSTUVWXYZ§±!@#$%^&*()_+";

    const columns = Math.floor(canvasWidth / 20);
    const rainDrops: any[] = [];

    for (let i = 0; i < columns; i++) {
      const x = i * 20;
      const char = characters[Math.floor(Math.random() * characters.length)];

      const raindrop = new (fabric as any).Text(char, {
        left: x,
        top: -20,
        fontSize: 16,
        fontFamily: "monospace",
        fill: color,
        opacity: Math.random() * 0.8 + 0.2,
        selectable: false,
        evented: false,
      });

      rainDrops.push(raindrop);
      this.canvas.add(raindrop);
    }

    // Animate rain drops
    const animateRain = () => {
      rainDrops.forEach((drop) => {
        drop.top += speed / 10;
        if (drop.top > canvasHeight) {
          drop.top = -20;
          drop.text = characters[Math.floor(Math.random() * characters.length)];
          drop.opacity = Math.random() * 0.8 + 0.2;
        }
      });

      this.canvas.renderAll();
    };

    const rainInterval = setInterval(animateRain, 100);
    this.activeEffects.set("matrix-rain", {
      interval: rainInterval,
      drops: rainDrops,
    });
  }

  // Glitch effect for stress testing
  async applyGlitchEffect(): Promise<void> {
    const objects = this.canvas.getObjects();

    objects.forEach((obj) => {
      if (Math.random() < 0.3) {
        // 30% chance to glitch each object
        const originalLeft = obj.left;
        const originalTop = obj.top;
        const originalOpacity = obj.opacity;

        // Random displacement
        obj.set({
          left: originalLeft + (Math.random() - 0.5) * 20,
          top: originalTop + (Math.random() - 0.5) * 20,
          opacity: originalOpacity * (0.5 + Math.random() * 0.5),
        });

        // Restore after brief moment
        setTimeout(
          () => {
            obj.set({
              left: originalLeft,
              top: originalTop,
              opacity: originalOpacity,
            });
            this.canvas.renderAll();
          },
          100 + Math.random() * 200,
        );
      }
    });

    this.canvas.renderAll();
  }

  // Neural network visualization effect
  async applyNeuralNetworkEffect(): Promise<void> {
    const nodes: any[] = [];
    const connections: any[] = [];

    // Create neural network nodes
    for (let i = 0; i < 20; i++) {
      const node = new (fabric as any).Circle({
        left: Math.random() * this.canvas.getWidth(),
        top: Math.random() * this.canvas.getHeight(),
        radius: 5 + Math.random() * 10,
        fill: "#00ff88",
        opacity: 0.7,
        selectable: false,
        evented: false,
      });

      nodes.push(node);
      this.canvas.add(node);
    }

    // Create connections between nearby nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const node1 = nodes[i];
        const node2 = nodes[j];

        const distance = Math.sqrt(
          Math.pow(node1.left - node2.left, 2) +
            Math.pow(node1.top - node2.top, 2),
        );

        if (distance < 150) {
          // Connect nearby nodes
          const connection = new (fabric as any).Line(
            [node1.left, node1.top, node2.left, node2.top],
            {
              stroke: "#00aaff",
              strokeWidth: 2,
              opacity: 0.3,
              selectable: false,
              evented: false,
            },
          );

          connections.push(connection);
          this.canvas.add(connection);
        }
      }
    }

    // Animate neural activity
    const animateNeural = () => {
      nodes.forEach((node) => {
        const newOpacity = 0.3 + Math.random() * 0.7;
        node.set("opacity", newOpacity);
      });

      connections.forEach((connection) => {
        const newOpacity = Math.random() * 0.5;
        connection.set("opacity", newOpacity);
      });

      this.canvas.renderAll();
    };

    const neuralInterval = setInterval(animateNeural, 200);
    this.activeEffects.set("neural-network", {
      interval: neuralInterval,
      nodes,
      connections,
    });
  }

  // Legal document highlight effect
  async applyLegalHighlightEffect(
    evidenceObjects: any[],
  ): Promise<void> {
    evidenceObjects.forEach((obj, index) => {
      const highlightColors = ["#ffff00", "#00ff00", "#ff6b6b", "#4ecdc4"];
      const color = highlightColors[index % highlightColors.length];

      // Create highlight glow
      const glow = new (fabric as any).Rect({
        left: obj.left - 5,
        top: obj.top - 5,
        width: obj.width + 10,
        height: obj.height + 10,
        fill: "transparent",
        stroke: color,
        strokeWidth: 3,
        opacity: 0,
        selectable: false,
        evented: false,
      });

      this.canvas.add(glow);

      // Animate highlight pulse
      const animateHighlight = () => {
        const targetOpacity = glow.opacity === 0 ? 0.8 : 0;
        (glow as any).animate({
          opacity: targetOpacity
        }, {
          duration: 1000,
          onChange: () => this.canvas.renderAll(),
          onComplete: animateHighlight,
        });
      };

      animateHighlight();
    });
  }

  // Performance stress test effect
  async applyStressTestEffect(): Promise<void> {
    const stressObjects: any[] = [];

    // Create many moving objects for stress testing
    for (let i = 0; i < 100; i++) {
      const shapes = ["circle", "rect", "triangle"];
      const shapeType = shapes[Math.floor(Math.random() * shapes.length)];

      let shape: any;

      switch (shapeType) {
        case "circle":
          shape = new (fabric as any).Circle({
            radius: 5 + Math.random() * 10,
            fill: `hsl(${Math.random() * 360}, 70%, 60%)`,
          });
          break;
        case "rect":
          shape = new (fabric as any).Rect({
            width: 10 + Math.random() * 20,
            height: 10 + Math.random() * 20,
            fill: `hsl(${Math.random() * 360}, 70%, 60%)`,
          });
          break;
        case "triangle":
          shape = new (fabric as any).Triangle({
            width: 10 + Math.random() * 20,
            height: 10 + Math.random() * 20,
            fill: `hsl(${Math.random() * 360}, 70%, 60%)`,
          });
          break;
      }

      shape.set({
        left: Math.random() * this.canvas.getWidth(),
        top: Math.random() * this.canvas.getHeight(),
        selectable: false,
        evented: false,
      });

      stressObjects.push(shape);
      this.canvas.add(shape);
    }

    // Animate all objects for stress testing
    const animateStress = () => {
      stressObjects.forEach((obj) => {
        obj.set({
          left: obj.left + (Math.random() - 0.5) * 10,
          top: obj.top + (Math.random() - 0.5) * 10,
          angle: obj.angle + (Math.random() - 0.5) * 20,
        });

        // Keep objects on canvas
        if (obj.left < 0 || obj.left > this.canvas.getWidth()) {
          obj.left = Math.random() * this.canvas.getWidth();
        }
        if (obj.top < 0 || obj.top > this.canvas.getHeight()) {
          obj.top = Math.random() * this.canvas.getHeight();
        }
      });

      this.canvas.renderAll();
    };

    const stressInterval = setInterval(animateStress, 16); // ~60 FPS
    this.activeEffects.set("stress-test", {
      interval: stressInterval,
      objects: stressObjects,
    });
  }

  // Particle system effect
  async applyParticleSystemEffect(): Promise<void> {
    const particles: any[] = [];
    const particleCount = 50;

    // Create particles
    for (let i = 0; i < particleCount; i++) {
      const particle = new (fabric as any).Circle({
        left: this.canvas.getWidth() / 2,
        top: this.canvas.getHeight() / 2,
        radius: 2 + Math.random() * 4,
        fill: `hsl(${120 + Math.random() * 120}, 80%, 60%)`,
        opacity: 0.8,
        selectable: false,
        evented: false,
      });

      // Add velocity properties
      (particle as any).vx = (Math.random() - 0.5) * 10;
      (particle as any).vy = (Math.random() - 0.5) * 10;
      (particle as any).life = 1.0;

      particles.push(particle);
      this.canvas.add(particle);
    }

    // Animate particle system
    const animateParticles = () => {
      particles.forEach((particle, index) => {
        // Update position
        particle.left += (particle as any).vx;
        particle.top += (particle as any).vy;

        // Apply gravity
        (particle as any).vy += 0.2;

        // Decrease life
        (particle as any).life -= 0.02;
        particle.opacity = (particle as any).life;

        // Reset particle if dead
        if ((particle as any).life <= 0) {
          particle.set({
            left: this.canvas.getWidth() / 2,
            top: this.canvas.getHeight() / 2,
            opacity: 0.8,
          });
          (particle as any).vx = (Math.random() - 0.5) * 10;
          (particle as any).vy = (Math.random() - 0.5) * 10;
          (particle as any).life = 1.0;
        }
      });

      this.canvas.renderAll();
    };

    const particleInterval = setInterval(animateParticles, 33); // ~30 FPS
    this.activeEffects.set("particles", {
      interval: particleInterval,
      particles,
    });
  }

  // Clear specific effect
  clearEffect(effectId: string): void {
    const effect = this.activeEffects.get(effectId);
    if (effect) {
      if (effect.interval) {
        clearInterval(effect.interval);
      }

      // Remove effect objects from canvas
      ["drops", "nodes", "connections", "objects", "particles"].forEach(
        (prop) => {
          if (effect[prop]) {
            effect[prop].forEach((obj: any) => {
              this.canvas.remove(obj);
            });
          }
        },
      );

      this.activeEffects.delete(effectId);
      this.canvas.renderAll();
    }
  }

  // Clear all effects
  clearAllEffects(): void {
    this.activeEffects.forEach((_, effectId) => {
      this.clearEffect(effectId);
    });
  }

  // Get active effects list
  getActiveEffects(): string[] {
    return Array.from(this.activeEffects.keys());
  }
}

// Pre-defined effect configurations
export const NEURAL_SPRITE_EFFECTS: SpriteEffect[] = [
  {
    id: "crt-scanlines",
    name: "CRT Scanlines",
    description: "Retro CRT monitor scanline effect",
    apply: async (canvas, options) => {
      const effects = new NeuralSpriteEffects(canvas);
      await effects.applyCRTScanlines(options?.intensity || 0.3);
    },
  },
  {
    id: "matrix-rain",
    name: "Matrix Digital Rain",
    description: "Legal AI-themed digital rain effect",
    apply: async (canvas, options) => {
      const effects = new NeuralSpriteEffects(canvas);
      await effects.applyMatrixRain(
        options?.speed || 50,
        options?.color || "#00ff41",
      );
    },
  },
  {
    id: "neural-network",
    name: "Neural Network Visualization",
    description: "Animated neural network connections",
    apply: async (canvas, options) => {
      const effects = new NeuralSpriteEffects(canvas);
      await effects.applyNeuralNetworkEffect();
    },
  },
  {
    id: "glitch",
    name: "Glitch Effect",
    description: "Random digital glitch distortion",
    apply: async (canvas, options) => {
      const effects = new NeuralSpriteEffects(canvas);
      await effects.applyGlitchEffect();
    },
  },
  {
    id: "particles",
    name: "Particle System",
    description: "Dynamic particle effects with physics",
    apply: async (canvas, options) => {
      const effects = new NeuralSpriteEffects(canvas);
      await effects.applyParticleSystemEffect();
    },
  },
  {
    id: "stress-test",
    name: "Performance Stress Test",
    description: "Many moving objects for performance testing",
    apply: async (canvas, options) => {
      const effects = new NeuralSpriteEffects(canvas);
      await effects.applyStressTestEffect();
    },
  },
];
