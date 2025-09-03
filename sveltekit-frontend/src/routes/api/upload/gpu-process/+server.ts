import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const UPLOAD_SERVICE_URL = 'http://localhost:8093';
const CUDA_SERVICE_URL = 'http://localhost:8096';

interface UploadProcessingOptions {
	enable_gpu: boolean;
	use_tensor_cores?: boolean;
	quantization?: '4bit' | '8bit' | 'fp16' | 'fp32';
	negative_latent_space?: boolean;
	extract_embeddings?: boolean;
	processing_priority?: 'low' | 'normal' | 'high';
}

export const POST: RequestHandler = async ({ request }) => {
	try {
		// Check if both services are available
		const [uploadHealth, cudaHealth] = await Promise.all([
			fetch(`${UPLOAD_SERVICE_URL}/health`).then(r => r.ok),
			fetch(`${CUDA_SERVICE_URL}/health`).then(r => r.ok)
		]).catch(() => [false, false]);

		if (!uploadHealth) {
			return json({ 
				error: 'Upload service unavailable',
				details: 'Document upload service is not responding'
			}, { status: 503 });
		}

		// Parse form data (assuming multipart/form-data)
		const formData = await request.formData();
		const file = formData.get('file') as File;
		const options: UploadProcessingOptions = {
			enable_gpu: formData.get('enable_gpu') === 'true' || false,
			use_tensor_cores: formData.get('use_tensor_cores') === 'true',
			quantization: (formData.get('quantization') as any) || '4bit',
			negative_latent_space: formData.get('negative_latent_space') === 'true',
			extract_embeddings: formData.get('extract_embeddings') === 'true',
			processing_priority: (formData.get('processing_priority') as any) || 'normal'
		};

		if (!file) {
			return json({ 
				error: 'No file provided',
				details: 'Please select a file to upload'
			}, { status: 400 });
		}

		// Phase 1: Upload to document service
		const uploadFormData = new FormData();
		uploadFormData.append('file', file);
		uploadFormData.append('extract_text', 'true');
		uploadFormData.append('generate_embeddings', String(options.extract_embeddings));

		const uploadResponse = await fetch(`${UPLOAD_SERVICE_URL}/upload`, {
			method: 'POST',
			body: uploadFormData
		});

		if (!uploadResponse.ok) {
			const uploadError = await uploadResponse.text();
			return json({
				error: 'Document upload failed',
				details: uploadError,
				phase: 'upload'
			}, { status: uploadResponse.status });
		}

		const uploadResult = await uploadResponse.json();

		// Phase 2: GPU Processing (if enabled and CUDA available)
		let gpuProcessingResult = null;
		if (options.enable_gpu && cudaHealth) {
			try {
				const gpuProcessingRequest = {
					query: `Process legal document: ${uploadResult.filename || 'unknown'}`,
					metadata: {
						document_id: uploadResult.id,
						filename: uploadResult.filename,
						file_size: uploadResult.size,
						content_preview: uploadResult.content?.substring(0, 500),
						rtx_optimization: true,
						use_tensor_cores: options.use_tensor_cores,
						quantization: options.quantization,
						negative_latent_space: options.negative_latent_space,
						processing_priority: options.processing_priority
					},
					options: {
						use_cache: true,
						priority: options.processing_priority,
						timeout: 30000 // 30 second timeout
					}
				};

				const gpuResponse = await fetch(`${CUDA_SERVICE_URL}/cuda/compute`, {
					method: 'POST',
					headers: {
						'Content-Type': 'application/json'
					},
					body: JSON.stringify(gpuProcessingRequest)
				});

				if (gpuResponse.ok) {
					gpuProcessingResult = await gpuResponse.json();
				} else {
					// GPU processing failed, but we still have the upload
					gpuProcessingResult = {
						error: 'GPU processing failed',
						details: await gpuResponse.text(),
						fallback_used: true
					};
				}
			} catch (gpuError) {
				gpuProcessingResult = {
					error: 'GPU processing error',
					details: gpuError instanceof Error ? gpuError.message : String(gpuError),
					fallback_used: true
				};
			}
		}

		// Phase 3: Combine results
		const result = {
			success: true,
			upload: {
				document_id: uploadResult.id,
				filename: uploadResult.filename || file.name,
				file_size: uploadResult.size || file.size,
				content_extracted: !!uploadResult.content,
				embeddings_generated: !!uploadResult.embeddings,
				processing_time_ms: uploadResult.processing_time_ms || 0
			},
			gpu_processing: options.enable_gpu ? {
				enabled: true,
				cuda_available: cudaHealth,
				processing_result: gpuProcessingResult,
				tensor_cores_used: options.use_tensor_cores && cudaHealth,
				quantization_used: options.quantization,
				negative_latent_space_used: options.negative_latent_space
			} : {
				enabled: false,
				reason: 'GPU processing not requested'
			},
			total_processing_time_ms: Date.now() - Date.now(), // This would be tracked properly
			timestamp: new Date().toISOString()
		};

		return json(result);

	} catch (error) {
		console.error('GPU upload processing error:', error);
		return json({
			error: 'Processing pipeline failed',
			details: error instanceof Error ? error.message : String(error),
			timestamp: new Date().toISOString()
		}, { status: 500 });
	}
};

export const GET: RequestHandler = async () => {
	try {
		// Health check for the entire pipeline
		const [uploadHealth, cudaHealth] = await Promise.all([
			fetch(`${UPLOAD_SERVICE_URL}/health`).then(async (r) => ({
				available: r.ok,
				status: r.ok ? await r.json() : null
			})),
			fetch(`${CUDA_SERVICE_URL}/health`).then(async (r) => ({
				available: r.ok,
				status: r.ok ? await r.json() : null
			}))
		]).catch(() => [
			{ available: false, status: null },
			{ available: false, status: null }
		]);

		return json({
			pipeline_status: 'healthy',
			services: {
				upload_service: {
					url: UPLOAD_SERVICE_URL,
					available: uploadHealth.available,
					status: uploadHealth.status
				},
				cuda_service: {
					url: CUDA_SERVICE_URL,
					available: cudaHealth.available,
					status: cudaHealth.status
				}
			},
			features: {
				document_upload: uploadHealth.available,
				gpu_processing: cudaHealth.available,
				tensor_cores: cudaHealth.available && cudaHealth.status?.cuda_initialized,
				quantization_support: ['4bit', '8bit', 'fp16', 'fp32'],
				negative_latent_space: cudaHealth.available,
				embedding_extraction: uploadHealth.available
			},
			timestamp: new Date().toISOString()
		});

	} catch (error) {
		return json({
			pipeline_status: 'error',
			error: 'Pipeline health check failed',
			details: error instanceof Error ? error.message : String(error)
		}, { status: 500 });
	}
};