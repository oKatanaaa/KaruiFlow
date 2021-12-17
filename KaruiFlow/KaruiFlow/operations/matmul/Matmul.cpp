#include "Matmul.h"
#include "CPUKernels.h"


namespace karuiflow {

	std::string MatMul::getOpName() {
		return "MatMul";
	}

	Kernel* MatMul::instantiateKernel(std::vector<TensorSpecs> inputs) {
		Device* device = inputs[0].device;
		if (device->getDeviceName() == "cpu")
			return new MatMulNumpyKernel();

		if (device->getDeviceName() == "cuda")
			throw std::runtime_error("Cuda is not supported.");
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
		if (dimA == 3 and dimB == 3) {
			if (shapeA[0] != shapeB[0] || shapeA[2] != shapeB[1])
				throw InconsistentShapes(getOpName(), { shapeA, shapeB });

			shapeC.push_back(shapeA[0]);
			shapeC.push_back(shapeA[1]);
			shapeC.push_back(shapeB[2]);
		}
		// Case 2
		else if (dimA == 3 && dimB == 2) {
			if (shapeA[2] != shapeB[0])
				throw InconsistentShapes(getOpName(), { shapeA, shapeB });

			shapeC.push_back(shapeA[0]);
			shapeC.push_back(shapeA[1]);
			shapeC.push_back(shapeB[1]);
		}
		// Case 3
		else if (dimA == 2 && dimB == 3) {
			if (shapeA[1] != shapeB[1])
				throw InconsistentShapes(getOpName(), { shapeA, shapeB });

			shapeC.push_back(shapeB[0]);
			shapeC.push_back(shapeA[0]);
			shapeC.push_back(shapeB[2]);
		}
		// Case 4
		else if (dimA == 3 && dimB == 3) {
			if (shapeA[1] != shapeB[0])
				throw InconsistentShapes(getOpName(), { shapeA, shapeB });

			shapeC.push_back(shapeA[0]);
			shapeC.push_back(shapeB[1]);
		}
		else {
			throw UnsuppotedShapes(getOpName(), { shapeA, shapeB });
		}

		DType* dtype;
		if (inputs[0].dtype->getName() == "float32" || inputs[1].dtype->getName() == "float32")
			dtype = new Float32();
		else
			dtype = new Int32();

		return TensorSpecs{ dtype, shapeC, inputs[0].device };
	}
}