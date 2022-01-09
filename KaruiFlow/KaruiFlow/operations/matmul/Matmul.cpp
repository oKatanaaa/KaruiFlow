#include "Matmul.h"
#include "MatMulCPUKernels.h"
#include "MatMulCUDAKernels.h"
#include "../../core/headers/LoggingUtils.h"


namespace karuiflow {

	std::string MatMul::getOpName() {
		return "MatMul";
	}

	Kernel* MatMul::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new MatMulNumpyKernel();

		auto shapeA = inputs[0].shape;
		auto shapeB = inputs[1].shape;
		if (device->getDeviceName() == "cuda" && 
			inputs[0].shape.size() == 2 && 
			inputs[0].shape.size() == 2 && 
			inputs[0].dtype->getName() == "float32")
			return new MatMulCudaKernel();
		else if (device->getDeviceName() == "cuda" &&
			inputs[0].shape.size() == 2 &&
			inputs[0].shape.size() == 2 &&
			inputs[0].dtype->getName() != "float32")
			throwException("No cuda kernel for operation supports data with dtype "
				+ inputs[0].dtype->getName() +
				". Please use float32 data or perform computation on CPU."
			);
		else {
			int dimA = shapeA.size();
			int dimB = shapeB.size();
			std::string msg = "No cuda kernel supports tensors with dims ";
			msg += std::to_string(dimA) + " and ";
			msg += std::to_string(dimB) + ". Only tensors of dim 2 are supported. ";
			msg += "Please perform this operation on CPU if you need to multiply tensors with higher dimensionality.";
			throwException(msg);
		}
	}

	TensorSpecs MatMul::inferOutputTensorSpecs(std::vector<TensorSpecs> inputs) {
		// A * B = C
		Shape shapeA = inputs[0].shape;
		Shape shapeB = inputs[1].shape;
		Shape shapeC;

		int dimA = shapeA.size();
		int dimB = shapeB.size();

		/*
		* There are several cases for shapes:
		* 1. [bs, n, m] [bs, m, k] = [bs, n, k]
		* 2. [bs, n, m] [m, k]     = [bs, n, k]
		* 3. [n, m]     [bs, m, k] = [bs, n, k]
		* 4. [n, m]     [m, k]     = [n, k]
		* 
		* Other cases are not supported.
		*/
		
		// Case 1
		if (dimA == 3 && dimB == 3) {
			if (shapeA[0] != shapeB[0] || shapeA[2] != shapeB[1])
				throw std::runtime_error(InconsistentShapes(getOpName(), { shapeA, shapeB }).what());

			shapeC.push_back(shapeA[0]);
			shapeC.push_back(shapeA[1]);
			shapeC.push_back(shapeB[2]);
		}
		// Case 2
		else if (dimA == 3 && dimB == 2) {
			if (shapeA[2] != shapeB[0])
				throw KF_ERROR(InconsistentShapes(getOpName(), { shapeA, shapeB }));

			shapeC.push_back(shapeA[0]);
			shapeC.push_back(shapeA[1]);
			shapeC.push_back(shapeB[1]);
		}
		// Case 3
		else if (dimA == 2 && dimB == 3) {
			if (shapeA[1] != shapeB[1])
				throw KF_ERROR(InconsistentShapes(getOpName(), { shapeA, shapeB }));

			shapeC.push_back(shapeB[0]);
			shapeC.push_back(shapeA[0]);
			shapeC.push_back(shapeB[2]);
		}
		// Case 4
		else if (dimA == 2 && dimB == 2) {
			if (shapeA[1] != shapeB[0])
				throw KF_ERROR(InconsistentShapes(getOpName(), { shapeA, shapeB }));

			shapeC.push_back(shapeA[0]);
			shapeC.push_back(shapeB[1]);
		}
		else {
			throw KF_ERROR(UnsuppotedShapes(getOpName(), { shapeA, shapeB }));
		}

		DType* dtype;
		if (inputs[0].dtype->getName() == "float32" || inputs[1].dtype->getName() == "float32")
			dtype = new Float32();
		else
			dtype = new Int32();

		return TensorSpecs{ dtype, shapeC, inputs[0].device };
	}
}