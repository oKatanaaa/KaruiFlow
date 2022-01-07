#include <spdlog/spdlog.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cutensor.h>

namespace karuiflow {

	void setDebugLogLevel() {
		spdlog::set_level(spdlog::level::debug);
	}

	void setInfoLogLevel() {
		spdlog::set_level(spdlog::level::info);
	}

	void setErrLogLevel() {
		spdlog::set_level(spdlog::level::err);
	}

	void setWarnLogLevel() {
		spdlog::set_level(spdlog::level::warn);
	}

	std::string shapeToString(std::vector<int>& shape) {
		std::string str;
		str += "[";
		for (int i = 0; i < shape.size(); i++) {
			str += std::to_string(shape[i]);
			if (i + 1 < shape.size())
				str += ", ";
		}
		str += "]";
		return str;
	}

	const char* cublasGetErrorString(cublasStatus_t status)
	{
		switch (status)
		{
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
		}
		return "Unknown error in CUBLAS";
	}

	void CudaCheck(cudaError_t error, const char* file, int line) {
		if (error != cudaSuccess)
		{
			fprintf(stderr, "Error: %s:%d, ", file, line);
			fprintf(stderr, "code: %d, reason: %s\n", error,
				cudaGetErrorString(error));
			throw std::runtime_error("Error in CUDA: " + std::string(cudaGetErrorString(error)));
		}
	}

	void CuTensorCheck(cutensorStatus_t error, const char* file, int line) {
		if (error != CUTENSOR_STATUS_SUCCESS)
		{
			fprintf(stderr, "Error: %s:%d, ", file, line);
			fprintf(stderr, "code: %d, reason: %s\n", error,
				cutensorGetErrorString(error));
			throw std::runtime_error("Error in CUDA: " + std::string(cutensorGetErrorString(error)));
		}
	}

	void CublasCheck(cublasStatus_t error, const char* file, int line) {
		if (error != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "Error: %s:%d, ", file, line);
			fprintf(stderr, "code: %d, reason: %s\n", error,
				cublasGetErrorString(error));
			throw std::runtime_error("Error in CUBLAS: " + std::string(cublasGetErrorString(error)));
		}
	}
}
