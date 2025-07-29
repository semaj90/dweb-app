import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import { TechStack, ProjectType } from './types';

export class StackAnalyzer {
    private workspaceRoot: string;

    constructor(workspaceRoot: string) {
        this.workspaceRoot = workspaceRoot;
    }

    /**
     * Analyze the entire tech stack from workspace files, dependencies, and configurations
     */
    async analyzeFullStack(): Promise<{ projectType: ProjectType; detectedStack: TechStack }> {
        const detectedStack: TechStack = {
            frontend: [],
            backend: [],
            databases: [],
            cloud: [],
            aiml: [],
            gpu: [],
            embedded: [],
            systems: [],
            scientific: [],
            gaming: [],
            mobile: [],
            web3: []
        };

        // Analyze package.json and dependencies
        await this.analyzePackageJson(detectedStack);
        
        // Analyze source files and extensions
        await this.analyzeSourceFiles(detectedStack);
        
        // Analyze configuration files
        await this.analyzeConfigFiles(detectedStack);
        
        // Analyze build and project files
        await this.analyzeBuildFiles(detectedStack);
        
        // Analyze CUDA and GPU specific files
        await this.analyzeGPUFiles(detectedStack);
        
        // Analyze embedded/hardware files
        await this.analyzeEmbeddedFiles(detectedStack);

        // Determine primary project type based on detected stack
        const projectType = this.determineProjectType(detectedStack);

        return { projectType, detectedStack };
    }

    /**
     * Analyze package.json for JavaScript/Node.js dependencies
     */
    private async analyzePackageJson(stack: TechStack): Promise<void> {
        try {
            const packageJsonPath = path.join(this.workspaceRoot, 'package.json');
            if (fs.existsSync(packageJsonPath)) {
                const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
                const allDeps = { ...packageJson.dependencies, ...packageJson.devDependencies };

                // Frontend frameworks
                if (allDeps['react'] || allDeps['@types/react']) stack.frontend.push('react');
                if (allDeps['vue'] || allDeps['@vue/cli']) stack.frontend.push('vue');
                if (allDeps['svelte'] || allDeps['@sveltejs/kit']) stack.frontend.push('svelte');
                if (allDeps['@angular/core']) stack.frontend.push('angular');
                if (allDeps['react-native']) stack.mobile.push('react-native');
                if (allDeps['flutter']) stack.mobile.push('flutter');

                // Backend
                if (allDeps['express'] || allDeps['fastify'] || allDeps['koa']) stack.backend.push('node');
                
                // Databases
                if (allDeps['pg'] || allDeps['postgresql']) stack.databases.push('postgresql');
                if (allDeps['mysql'] || allDeps['mysql2']) stack.databases.push('mysql');
                if (allDeps['mongodb'] || allDeps['mongoose']) stack.databases.push('mongodb');
                if (allDeps['redis'] || allDeps['ioredis']) stack.databases.push('redis');
                if (allDeps['neo4j-driver']) stack.databases.push('neo4j');

                // AI/ML
                if (allDeps['@tensorflow/tfjs'] || allDeps['tensorflow']) stack.aiml.push('tensorflow');
                if (allDeps['torch'] || allDeps['pytorch']) stack.aiml.push('pytorch');
                if (allDeps['opencv-js'] || allDeps['opencv4nodejs']) stack.aiml.push('opencv');
                if (allDeps['@huggingface/transformers']) stack.aiml.push('transformers');
                if (allDeps['langchain']) stack.aiml.push('langchain');
                if (allDeps['ollama']) stack.aiml.push('ollama');

                // Cloud & DevOps
                if (allDeps['aws-sdk'] || allDeps['@aws-sdk/client-s3']) stack.cloud.push('aws');
                if (allDeps['@google-cloud/storage']) stack.cloud.push('gcp');
                if (allDeps['@azure/storage-blob']) stack.cloud.push('azure');

                // Scientific Computing
                if (allDeps['numpy'] || allDeps['pandas']) stack.scientific.push('numpy');
                if (allDeps['jupyter']) stack.scientific.push('jupyter');

                // Web3/Blockchain
                if (allDeps['ethers'] || allDeps['web3']) stack.web3.push('ethereum');
                if (allDeps['hardhat']) stack.web3.push('hardhat');
                if (allDeps['truffle']) stack.web3.push('truffle');

                // Gaming
                if (allDeps['three'] || allDeps['babylonjs']) stack.gaming.push('opengl');
            }
        } catch (error) {
            console.log('Error analyzing package.json:', error);
        }
    }

    /**
     * Analyze source files by extensions and content
     */
    private async analyzeSourceFiles(stack: TechStack): Promise<void> {
        const workspaceFiles = await vscode.workspace.findFiles('**/*', '**/node_modules/**', 1000);
        
        for (const file of workspaceFiles) {
            const filePath = file.fsPath;
            const extension = path.extname(filePath).toLowerCase();
            const fileName = path.basename(filePath).toLowerCase();

            // Language detection by extension
            switch (extension) {
                case '.py':
                    if (!stack.backend.includes('python')) stack.backend.push('python');
                    await this.analyzePythonFile(filePath, stack);
                    break;
                case '.rs':
                    if (!stack.backend.includes('rust')) stack.backend.push('rust');
                    break;
                case '.go':
                    if (!stack.backend.includes('go')) stack.backend.push('go');
                    break;
                case '.java':
                    if (!stack.backend.includes('java')) stack.backend.push('java');
                    break;
                case '.cpp':
                case '.cc':
                case '.cxx':
                    if (!stack.backend.includes('cpp')) stack.backend.push('cpp');
                    await this.analyzeCppFile(filePath, stack);
                    break;
                case '.c':
                    if (!stack.backend.includes('c')) stack.backend.push('c');
                    await this.analyzeCFile(filePath, stack);
                    break;
                case '.cs':
                    if (!stack.backend.includes('csharp')) stack.backend.push('csharp');
                    break;
                case '.cu':
                case '.cuh':
                    if (!stack.gpu.includes('cuda')) stack.gpu.push('cuda');
                    break;
                case '.cl':
                    if (!stack.gpu.includes('opencl')) stack.gpu.push('opencl');
                    break;
                case '.v':
                    if (!stack.embedded.includes('verilog')) stack.embedded.push('verilog');
                    break;
                case '.vhd':
                case '.vhdl':
                    if (!stack.embedded.includes('vhdl')) stack.embedded.push('vhdl');
                    break;
                case '.ino':
                    if (!stack.embedded.includes('arduino')) stack.embedded.push('arduino');
                    break;
                case '.m':
                    if (fileName.includes('matlab') || this.containsMatlabSyntax(filePath)) {
                        if (!stack.scientific.includes('matlab')) stack.scientific.push('matlab');
                    }
                    break;
                case '.r':
                    if (!stack.scientific.includes('r-lang')) stack.scientific.push('r-lang');
                    break;
                case '.ipynb':
                    if (!stack.scientific.includes('jupyter')) stack.scientific.push('jupyter');
                    break;
                case '.sol':
                    if (!stack.web3.includes('solidity')) stack.web3.push('solidity');
                    break;
                case '.unity':
                case '.unitypackage':
                    if (!stack.gaming.includes('unity')) stack.gaming.push('unity');
                    break;
                case '.uproject':
                    if (!stack.gaming.includes('unreal')) stack.gaming.push('unreal');
                    break;
            }

            // Special file name patterns
            if (fileName.includes('dockerfile')) {
                if (!stack.cloud.includes('docker')) stack.cloud.push('docker');
            }
            if (fileName.includes('makefile') || fileName === 'cmake') {
                // Could indicate systems programming
                if (!stack.systems.includes('bare-metal')) stack.systems.push('bare-metal');
            }
        }
    }

    /**
     * Analyze Python files for specific libraries and frameworks
     */
    private async analyzePythonFile(filePath: string, stack: TechStack): Promise<void> {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            
            // AI/ML libraries
            if (content.includes('import tensorflow') || content.includes('import tf')) {
                if (!stack.aiml.includes('tensorflow')) stack.aiml.push('tensorflow');
            }
            if (content.includes('import torch') || content.includes('import pytorch')) {
                if (!stack.aiml.includes('pytorch')) stack.aiml.push('pytorch');
            }
            if (content.includes('import cv2') || content.includes('import opencv')) {
                if (!stack.aiml.includes('opencv')) stack.aiml.push('opencv');
            }
            if (content.includes('from transformers')) {
                if (!stack.aiml.includes('transformers')) stack.aiml.push('transformers');
            }

            // Scientific computing
            if (content.includes('import numpy') || content.includes('import np')) {
                if (!stack.scientific.includes('numpy')) stack.scientific.push('numpy');
            }
            if (content.includes('import pandas') || content.includes('import pd')) {
                if (!stack.scientific.includes('pandas')) stack.scientific.push('pandas');
            }
            if (content.includes('import scipy')) {
                if (!stack.scientific.includes('scipy')) stack.scientific.push('scipy');
            }

            // GPU computing
            if (content.includes('import cupy') || content.includes('from cupy')) {
                if (!stack.gpu.includes('cupy')) stack.gpu.push('cupy');
            }
            if (content.includes('import numba') || content.includes('@numba.jit')) {
                if (!stack.gpu.includes('numba')) stack.gpu.push('numba');
            }

            // CUDA specific imports
            if (content.includes('cuda') || content.includes('gpu')) {
                if (!stack.gpu.includes('cuda')) stack.gpu.push('cuda');
            }
        } catch (error) {
            console.log('Error analyzing Python file:', error);
        }
    }

    /**
     * Analyze C++ files for CUDA and system programming indicators
     */
    private async analyzeCppFile(filePath: string, stack: TechStack): Promise<void> {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            
            // CUDA includes
            if (content.includes('#include <cuda') || content.includes('#include <thrust') || 
                content.includes('__global__') || content.includes('__device__')) {
                if (!stack.gpu.includes('cuda')) stack.gpu.push('cuda');
                if (!stack.gpu.includes('nvidia-toolkit')) stack.gpu.push('nvidia-toolkit');
            }
            
            // Thrust library
            if (content.includes('#include <thrust')) {
                if (!stack.gpu.includes('thrust')) stack.gpu.push('thrust');
            }

            // OpenCL
            if (content.includes('#include <CL/cl.h>') || content.includes('opencl')) {
                if (!stack.gpu.includes('opencl')) stack.gpu.push('opencl');
            }

            // Graphics APIs
            if (content.includes('#include <GL/gl.h>') || content.includes('opengl')) {
                if (!stack.gaming.includes('opengl')) stack.gaming.push('opengl');
            }
            if (content.includes('#include <vulkan') || content.includes('VkInstance')) {
                if (!stack.gaming.includes('vulkan')) stack.gaming.push('vulkan');
            }

            // System programming indicators
            if (content.includes('#include <sys/') || content.includes('#include <linux/') ||
                content.includes('kernel') || content.includes('module_init')) {
                if (!stack.systems.includes('kernel')) stack.systems.push('kernel');
            }

            // Embedded indicators
            if (content.includes('gpio') || content.includes('interrupt') || 
                content.includes('register') || content.includes('volatile')) {
                if (!stack.embedded.includes('bare-metal')) stack.embedded.push('bare-metal');
            }
        } catch (error) {
            console.log('Error analyzing C++ file:', error);
        }
    }

    /**
     * Analyze C files for embedded and systems programming
     */
    private async analyzeCFile(filePath: string, stack: TechStack): Promise<void> {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            
            // Embedded indicators
            if (content.includes('#include <avr/') || content.includes('PORTA') || 
                content.includes('DDRB') || content.includes('ISR(')) {
                if (!stack.embedded.includes('arduino')) stack.embedded.push('arduino');
            }

            // ARM Cortex-M indicators
            if (content.includes('stm32') || content.includes('CMSIS') || 
                content.includes('__NVIC') || content.includes('cortex')) {
                if (!stack.embedded.includes('stm32')) stack.embedded.push('stm32');
            }

            // Real-time OS indicators
            if (content.includes('FreeRTOS') || content.includes('xTaskCreate') ||
                content.includes('vTaskDelay')) {
                if (!stack.systems.includes('rtos')) stack.systems.push('rtos');
            }

            // Raspberry Pi indicators
            if (content.includes('wiringPi') || content.includes('bcm2835') ||
                content.includes('/dev/mem')) {
                if (!stack.embedded.includes('raspberry-pi')) stack.embedded.push('raspberry-pi');
            }
        } catch (error) {
            console.log('Error analyzing C file:', error);
        }
    }

    /**
     * Analyze configuration files for technology indicators
     */
    private async analyzeConfigFiles(stack: TechStack): Promise<void> {
        const configFiles = [
            'docker-compose.yml', 'Dockerfile', 'kubernetes.yaml', 'k8s.yaml',
            'terraform.tf', 'ansible.yml', 'CMakeLists.txt', 'Makefile',
            'requirements.txt', 'Pipfile', 'poetry.lock', 'conda.yml'
        ];

        for (const configFile of configFiles) {
            const filePath = path.join(this.workspaceRoot, configFile);
            if (fs.existsSync(filePath)) {
                const content = fs.readFileSync(filePath, 'utf8');

                if (configFile.includes('docker')) {
                    if (!stack.cloud.includes('docker')) stack.cloud.push('docker');
                }
                if (configFile.includes('kubernetes') || configFile.includes('k8s')) {
                    if (!stack.cloud.includes('kubernetes')) stack.cloud.push('kubernetes');
                }
                if (configFile.includes('terraform')) {
                    if (!stack.cloud.includes('terraform')) stack.cloud.push('terraform');
                }
                if (configFile.includes('ansible')) {
                    if (!stack.cloud.includes('ansible')) stack.cloud.push('ansible');
                }

                // Analyze Python requirements
                if (configFile === 'requirements.txt') {
                    this.analyzePythonRequirements(content, stack);
                }
            }
        }
    }

    /**
     * Analyze Python requirements.txt for specific libraries
     */
    private analyzePythonRequirements(content: string, stack: TechStack): void {
        const lines = content.split('\n');
        for (const line of lines) {
            const packageName = line.split('==')[0].split('>=')[0].split('<=')[0].trim().toLowerCase();
            
            // AI/ML packages
            if (['tensorflow', 'tensorflow-gpu'].includes(packageName)) {
                if (!stack.aiml.includes('tensorflow')) stack.aiml.push('tensorflow');
            }
            if (['torch', 'pytorch'].includes(packageName)) {
                if (!stack.aiml.includes('pytorch')) stack.aiml.push('pytorch');
            }
            if (['opencv-python', 'cv2'].includes(packageName)) {
                if (!stack.aiml.includes('opencv')) stack.aiml.push('opencv');
            }
            if (packageName === 'transformers') {
                if (!stack.aiml.includes('transformers')) stack.aiml.push('transformers');
            }

            // GPU computing
            if (['cupy', 'cupy-cuda'].includes(packageName)) {
                if (!stack.gpu.includes('cupy')) stack.gpu.push('cupy');
            }
            if (packageName === 'numba') {
                if (!stack.gpu.includes('numba')) stack.gpu.push('numba');
            }

            // Scientific computing
            if (packageName === 'numpy') {
                if (!stack.scientific.includes('numpy')) stack.scientific.push('numpy');
            }
            if (packageName === 'pandas') {
                if (!stack.scientific.includes('pandas')) stack.scientific.push('pandas');
            }
            if (packageName === 'scipy') {
                if (!stack.scientific.includes('scipy')) stack.scientific.push('scipy');
            }
        }
    }

    /**
     * Analyze build files and project configurations
     */
    private async analyzeBuildFiles(stack: TechStack): Promise<void> {
        // Check for various build systems and project files
        const buildFiles = [
            'CMakeLists.txt', 'Makefile', 'build.gradle', 'pom.xml',
            'Cargo.toml', 'go.mod', 'composer.json', 'pubspec.yaml'
        ];

        for (const buildFile of buildFiles) {
            const filePath = path.join(this.workspaceRoot, buildFile);
            if (fs.existsSync(filePath)) {
                const content = fs.readFileSync(filePath, 'utf8');

                if (buildFile === 'Cargo.toml') {
                    if (!stack.backend.includes('rust')) stack.backend.push('rust');
                }
                if (buildFile === 'go.mod') {
                    if (!stack.backend.includes('go')) stack.backend.push('go');
                }
                if (buildFile === 'pubspec.yaml') {
                    if (!stack.mobile.includes('flutter')) stack.mobile.push('flutter');
                }
                if (['build.gradle', 'pom.xml'].includes(buildFile)) {
                    if (!stack.backend.includes('java')) stack.backend.push('java');
                }

                // Analyze CMakeLists.txt for CUDA
                if (buildFile === 'CMakeLists.txt') {
                    if (content.includes('find_package(CUDA)') || content.includes('enable_language(CUDA)')) {
                        if (!stack.gpu.includes('cuda')) stack.gpu.push('cuda');
                        if (!stack.gpu.includes('nvidia-toolkit')) stack.gpu.push('nvidia-toolkit');
                    }
                }
            }
        }
    }

    /**
     * Analyze GPU-specific files and configurations
     */
    private async analyzeGPUFiles(stack: TechStack): Promise<void> {
        // Look for NVIDIA/CUDA specific files
        const gpuFiles = await vscode.workspace.findFiles('**/*.{cu,cuh,ptx}', '**/node_modules/**', 100);
        if (gpuFiles.length > 0) {
            if (!stack.gpu.includes('cuda')) stack.gpu.push('cuda');
            if (!stack.gpu.includes('nvidia-toolkit')) stack.gpu.push('nvidia-toolkit');
        }

        // Check for CUDA toolkit installation indicators
        const cudaIndicators = [
            'nvcc_config.h', 'cuda_runtime.h', 'cublas.h', 'curand.h',
            'cudnn.h', 'tensorrt.h'
        ];

        for (const indicator of cudaIndicators) {
            const files = await vscode.workspace.findFiles(`**/${indicator}`, '**/node_modules/**', 10);
            if (files.length > 0) {
                if (!stack.gpu.includes('nvidia-toolkit')) stack.gpu.push('nvidia-toolkit');
                if (indicator.includes('tensorrt')) {
                    if (!stack.gpu.includes('tensorrt')) stack.gpu.push('tensorrt');
                }
            }
        }
    }

    /**
     * Analyze embedded/hardware specific files
     */
    private async analyzeEmbeddedFiles(stack: TechStack): Promise<void> {
        // Arduino files
        const arduinoFiles = await vscode.workspace.findFiles('**/*.ino', '**/node_modules/**', 100);
        if (arduinoFiles.length > 0) {
            if (!stack.embedded.includes('arduino')) stack.embedded.push('arduino');
        }

        // Verilog/VHDL files
        const hdlFiles = await vscode.workspace.findFiles('**/*.{v,vh,vhd,vhdl}', '**/node_modules/**', 100);
        if (hdlFiles.length > 0) {
            for (const file of hdlFiles) {
                const ext = path.extname(file.fsPath).toLowerCase();
                if (['.v', '.vh'].includes(ext)) {
                    if (!stack.embedded.includes('verilog')) stack.embedded.push('verilog');
                } else if (['.vhd', '.vhdl'].includes(ext)) {
                    if (!stack.embedded.includes('vhdl')) stack.embedded.push('vhdl');
                }
            }
        }

        // Check for embedded Linux indicators
        const embeddedLinuxFiles = ['buildroot', 'yocto', 'device-tree', 'kernel-config'];
        for (const indicator of embeddedLinuxFiles) {
            const files = await vscode.workspace.findFiles(`**/*${indicator}*`, '**/node_modules/**', 10);
            if (files.length > 0) {
                if (!stack.systems.includes('embedded-linux')) stack.systems.push('embedded-linux');
            }
        }
    }

    /**
     * Check if file contains MATLAB syntax
     */
    private containsMatlabSyntax(filePath: string): boolean {
        try {
            const content = fs.readFileSync(filePath, 'utf8');
            return content.includes('function ') && (content.includes('end') || content.includes('%'));
        } catch {
            return false;
        }
    }

    /**
     * Determine the primary project type based on detected technologies
     */
    private determineProjectType(stack: TechStack): ProjectType {
        // GPU/CUDA Computing
        if (stack.gpu.length > 0 && (stack.gpu.includes('cuda') || stack.gpu.includes('nvidia-toolkit'))) {
            return 'cuda-gpu-computing';
        }

        // Embedded/Electrical Engineering
        if (stack.embedded.length > 0 || stack.systems.length > 0) {
            return 'electrical-embedded';
        }

        // AI/ML Research
        if (stack.aiml.length >= 2 && stack.scientific.length > 0) {
            return 'ml-ai-research';
        }

        // Scientific Computing
        if (stack.scientific.length >= 2) {
            return 'scientific-computing';
        }

        // Systems Programming
        if (stack.systems.length > 0 || (stack.backend.includes('c') || stack.backend.includes('cpp'))) {
            return 'systems-programming';
        }

        // Web3/Blockchain
        if (stack.web3.length > 0) {
            return 'web-fullstack'; // Could be specialized further
        }

        // Mobile Development
        if (stack.mobile.length > 0) {
            return 'mobile-app';
        }

        // Gaming
        if (stack.gaming.length > 0) {
            return 'desktop-native';
        }

        // Web Development (SvelteKit Legal AI specifically)
        if (stack.frontend.includes('svelte') && stack.databases.includes('postgresql')) {
            return 'sveltekit-legal-ai';
        }

        // General Web Development
        if (stack.frontend.length > 0 && stack.backend.length > 0) {
            if (stack.frontend.includes('react')) return 'react-nextjs';
            if (stack.frontend.includes('vue')) return 'vue-nuxt';
            return 'web-fullstack';
        }

        return 'generic';
    }

    /**
     * Get MCP documentation suggestions based on detected stack
     */
    getMCPDocSuggestions(stack: TechStack): Array<{library: string; topics: string[]}> {
        const suggestions: Array<{library: string; topics: string[]}> = [];

        // GPU/CUDA suggestions
        if (stack.gpu.includes('cuda')) {
            suggestions.push({
                library: 'cuda',
                topics: ['programming-guide', 'best-practices', 'optimization', 'memory-management']
            });
        }
        if (stack.gpu.includes('nvidia-toolkit')) {
            suggestions.push({
                library: 'nvidia-toolkit',
                topics: ['installation', 'cuDNN', 'tensorRT', 'profiling']
            });
        }

        // AI/ML suggestions
        if (stack.aiml.includes('pytorch')) {
            suggestions.push({
                library: 'pytorch',
                topics: ['tensors', 'neural-networks', 'gpu-acceleration', 'distributed-training']
            });
        }
        if (stack.aiml.includes('tensorflow')) {
            suggestions.push({
                library: 'tensorflow',
                topics: ['eager-execution', 'keras', 'tensorboard', 'serving']
            });
        }

        // Systems programming suggestions
        if (stack.systems.includes('kernel')) {
            suggestions.push({
                library: 'linux-kernel',
                topics: ['module-development', 'device-drivers', 'debugging', 'memory-management']
            });
        }

        // Embedded suggestions
        if (stack.embedded.includes('arduino')) {
            suggestions.push({
                library: 'arduino',
                topics: ['libraries', 'interrupts', 'serial-communication', 'sensors']
            });
        }

        // Scientific computing suggestions
        if (stack.scientific.includes('numpy')) {
            suggestions.push({
                library: 'numpy',
                topics: ['arrays', 'linear-algebra', 'broadcasting', 'performance']
            });
        }

        return suggestions;
    }
}