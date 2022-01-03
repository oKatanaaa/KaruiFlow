/*
 * Copyright (c) 2019-21, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


/**
 * @file
 * @brief This file contains all public function declarations of the cuTENSOR
 * library.
 */
#pragma once

#define CUTENSOR_MAJOR 1 //!< cuTensor major version.
#define CUTENSOR_MINOR 4 //!< cuTensor minor version.
#define CUTENSOR_PATCH 0 //!< cuTensor patch version.
#define CUTENSOR_VERSION (CUTENSOR_MAJOR * 10000 + CUTENSOR_MINOR * 100 + CUTENSOR_PATCH)

#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include <cutensor/types.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \mainpage cuTENSOR: A High-Performance CUDA Library for Tensor Primitives
 *
 * \section intro Introduction
 *
 * \subsection nomen Nomenclature
 *
 * The term tensor refers to an \b order-n (a.k.a.,
 * n-dimensional) array. One can think of tensors as a generalization of
 * matrices to higher \b orders.

 * For example, scalars, vectors, and matrices are
 * order-0, order-1, and order-2 tensors, respectively.
 *
 * An order-n tensor has n \b modes. Each mode has an \b extent (a.k.a. size).
 * Each mode you can specify a \b stride s > 0. This \b stride
 * describes offset of two logically consecutive elements in physical (i.e., linearized) memory.
 * This is similar to the leading-dimension in BLAS.

 * cuTENSOR, by default, adheres to a generalized \b column-major data layout.
 * For example: \f$A_{a,b,c} \in {R}^{4\times 8 \times 12}\f$
 * is an order-3 tensor with the extent of the a-mode, b-mode, and c-mode
 * respectively being 4, 8, and 12. If not explicitly specified, the strides are
 * assumed to be: stride(a) = 1, stride(b) = extent(a), stride(c) = extent(a) *
 * extent(b).

 * For a general order-n tensor \f$A_{i_1,i_2,...,i_n}\f$ we require that the strides do
 * not lead to overlapping memory accesses; for instance, \f$stride(i_1) \geq 1\f$, and
 * \f$stride(i_l) \geq stride(i_{l-1}) * extent(i_{l-1})\f$.

 * We say that a tensor is \b packed if it is contiguously stored in memory along all
 * modes. That is, \f$ stride(i_1) = 1\f$ and \f$stride(i_l) =stride(i_{l-1}) *
 * extent(i_{l-1})\f$).
 *
 * \subsection einsum Einstein Notation
 * We adhere to the "Einstein notation": Modes that appear in the input
 * tensors, and that do not appear in the output tensor, are implicitly
 * contracted.
 *
 * \section api API Reference
 * For details on the API please refer to \ref cutensor.h and \ref types.h.
 *
 */

/**
 * \brief Initializes the cuTENSOR library
 *
 * \details The device associated with a particular cuTENSOR handle is assumed to remain
 * unchanged after the cutensorInit() call. In order for the cuTENSOR library to 
 * use a different device, the application must set the new device to be used by
 * calling cudaSetDevice() and then create another cuTENSOR handle, which will
 * be associated with the new device, by calling cutensorInit().
 *
 * \param[out] handle Pointer to cutensorHandle_t
 *
 * \returns CUTENSOR_STATUS_SUCCESS on success and an error code otherwise
 * \remarks blocking, no reentrant, and thread-safe
 */
cutensorStatus_t cutensorInit(cutensorHandle_t* handle);

/**
 * \brief Detaches cachelines from cache (beta feature).
 *
 * Detaches cachelines from cache (i.e., releases the owner ship of the attached cache
 * lines back to the caller) and deallocates any data structures that have been allocated
 * as part of cutensorHandleAttachPlanCachelines().
 *
 * This function is not thread-safe.
 *
 * \param[in,out] handle Opaque handle holding cuTENSOR's library context. The cachelines corresponding to this cache will be detached; after
 * this call the user again takes full ownership over the chacheline buffer.
 *
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED e.g., if no cachelines had been attached
 * \remarks non-blocking, no reentrant, and not thread-safe
 */
cutensorStatus_t cutensorHandleDetachPlanCachelines(cutensorHandle_t* handle);

/**
 * \brief Attaches cachelines to the plan cache (beta feature).
 *
 * This function attaches the cachelines to the handle and allocates some internal data
 * structures required for the cache; hence, it is critical that users also call
 * cutensorHandleDetachPlanCachelines() to free those resources again.
 *
 * The handle assumes ownership over the attached cachelines stay valid --at least-- until
 * cutensorHandleDetachPlanCachelines has been called.  Moreover, the attached cachelines
 * must not be shared with other handles (i.e., after this call they are assumed to be
 * exclusively used by the handle).
 *
 * While this function is not thread-safe, the resulting cache can be shared across
 * different threads in a thread-safe manner.
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context. The cachelines will be attached to the handle; the cachelines must
 * remain valid until they have been detached (see cutensorPlanCacheDetachCachelines)
 * \param[in] cachelines array of user-allocated cachelines (host memory). 
 * \param[in] numCachelines Number of provided cachelines.
 *
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \remarks non-blocking, no reentrant, and not thread-safe
 */
cutensorStatus_t cutensorHandleAttachPlanCachelines(cutensorHandle_t* handle,
                                             cutensorPlanCacheline_t cachelines[],
                                             const uint32_t numCachelines);

/**
 * \brief Writes the attached Plan-Cache to file (beta feature).
 *
 * This function is thread-safe.
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] filename Specifies the filename (including the absolute path) to the file
 * that should hold all the cache information. Warning: an existing file will be
 * overwritten.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_INVALID_VALUE if the no cache has been attached
 * \retval CUTENSOR_STATUS_IO_ERROR if the file cannot be written to
 *
 * \remarks non-blocking, no reentrant, and thread-safe
 */
cutensorStatus_t cutensorHandleWriteCacheToFile(const cutensorHandle_t* handle,
                                                const char filename[]);

/**
 * \brief Reads a Plan-Cache from file and overwrites the attached cachelines (beta feature).
 *
 * A cache is only valid for the same cuTENSOR version and CUDA version; moreover, the
 * GPU architecture (incl. multiprocessor count) must match, otherwise
 * CUTENSOR_STATUS_INVALID_VALUE will be returned.
 *
 * It's important that the user already attached sufficient cachelines (via cutensorHandleAttachPlanCachelines),
 * otherwise CUTENSOR_STATUS_INVALID_VALUE will be returned.
 *
 * This function is thread-safe.
 *
 * \param[in,out] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] filename Specifies the filename (including the absolute path) to the file
 * that holds all the cache information that have previously been written by cutensorHandleWriteCacheToFile().
 * \param[out] numCachelinesRead On exit, this variable will hold the number of
 * successfully-read cachelines, if CUTENSOR_STATUS_SUCCESS is returned. Otherwise, this
 * variable will hold the number of cachelines that are required to read all
 * cachelines associated to the cache pointed to by `filename`; in that case
 * CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE is returned.
 *
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if the stored cache was created by a different
 * cuTENSOR- or CUDA-version or if the GPU architecture (incl. multiprocessor count) doesn't match
 * \retval CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE if the stored cache requires more
 * cachelines than those that are currently attached to the handle
 * \retval CUTENSOR_STATUS_IO_ERROR if the file cannot be read
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 *
 * \remarks non-blocking, no reentrant, and thread-safe
 */
cutensorStatus_t cutensorHandleReadCacheFromFile(cutensorHandle_t* handle,
                                                 const char filename[],
                                                 uint32_t* numCachelinesRead);

/**
 * \brief Initializes a tensor descriptor
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[out] desc Pointer to the address where the allocated tensor descriptor object is stored.
 * \param[in] numModes Number of modes.
 * \param[in] extent Extent of each mode (must be larger than zero).
 * \param[in] stride stride[i] denotes the displacement (stride) between two consecutive elements in the ith-mode.
 *            If stride is NULL, a packed generalized column-major memory
 *            layout is assumed (i.e., the strides increase monotonically from left to
 *            right). Each stride must be larger than zero; to be precise, a stride of zero can be
 *            achieved by omitting this mode entirely; for instance instead of writing
 *            C[a,b] = A[b,a] with strideA(a) = 0, you can write C[a,b] = A[b] directly;
 *            cuTENSOR will then automatically infer that the a-mode in A should be broadcasted).
 * \param[in] dataType Data type of the stored entries.
 * \param[in] unaryOp Unary operator that will be applied to each element of the corresponding
 *            tensor in a lazy fashion (i.e., the algorithm uses this tensor as its operand only once).
 *            The original data of this tensor remains unchanged.
 * \pre extent and stride arrays must each contain at least sizeof(int64_t) * numModes bytes
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED if the requested descriptor is not supported (e.g., due to non-supported data type).
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 * \remarks non-blocking, no reentrant, and thread-safe
 */
cutensorStatus_t cutensorInitTensorDescriptor(const cutensorHandle_t* handle,
                                              cutensorTensorDescriptor_t* desc,
                                              const uint32_t numModes,
                                              const int64_t extent[],
                                              const int64_t stride[],
                                              cudaDataType_t dataType,
                                              cutensorOperator_t unaryOp);

/**
 * \brief Element-wise tensor operation with three inputs
 *
 * \details This function performs a element-wise tensor operation of the form:
 * \f[ D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{ABC}(\Phi_{AB}(\alpha \Psi_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \beta \Psi_B(B_{\Pi^B(i_0,i_1,...,i_n)})), \gamma \Psi_C(C_{\Pi^C(i_0,i_1,...,i_n)})) \f]
 *
 * Where
 *    - A,B,C,D are multi-mode tensors (of arbitrary data types).
 *    - \f$\Pi^A, \Pi^B, \Pi^C \f$ are permutation operators that permute the modes of A, B, and C respectively.
 *    - \f$\Psi_{A},\Psi_{B},\Psi_{C}\f$ are unary element-wise operators (e.g., IDENTITY, CONJUGATE).
 *    - \f$\Phi_{ABC}, \Phi_{AB}\f$ are binary element-wise operators (e.g., ADD, MUL, MAX, MIN).
 *
 * Notice that the broadcasting (of a mode) can be achieved by simply omitting that mode from the respective tensor.
 *
 * Moreover, modes may appear in any order, giving users a greater flexibility. The only <b>restrictions</b> are:
 *    - modes that appear in A or B _must_ also appear in the output tensor; a mode that only appears in the input would be contracted and such an operation would be covered by either cutensorContraction or cutensorReduction.
 *    - each mode may appear in each tensor at most once.
 *
 * Input tensors may be read even if the value
 * of the corresponding scalar is zero.
 *
 * Examples:
 *    - \f$ D_{a,b,c,d} = A_{b,d,a,c}\f$
 *    - \f$ D_{a,b,c,d} = 2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a}\f$
 *    - \f$ D_{a,b,c,d} = 2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a} + C_{a,b,c,d}\f$
 *    - \f$ D_{a,b,c,d} = min((2.2 * A_{b,d,a,c} + 1.3 * B_{c,b,d,a}), C_{a,b,c,d})\f$
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +---------------+---------------+---------------+---------------+
 * |     typeA     |     typeB     |     typeC     |  typeScalar   |
 * +===============+===============+===============+===============+
 * |  CUDA_R_16F   |  CUDA_R_16F   |  CUDA_R_16F   |  CUDA_R_16F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_R_16F   |  CUDA_R_16F   |  CUDA_R_16F   |  CUDA_R_32F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_R_16BF  |  CUDA_R_16BF  |  CUDA_R_16BF  |  CUDA_R_16BF  |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_R_16BF  |  CUDA_R_16BF  |  CUDA_R_16BF  |  CUDA_R_32F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_R_32F   |  CUDA_R_32F   |  CUDA_R_32F   |  CUDA_R_32F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_R_64F   |  CUDA_R_64F   |  CUDA_R_64F   |  CUDA_R_64F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_C_32F   |  CUDA_C_32F   |  CUDA_C_32F   |  CUDA_C_32F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_C_64F   |  CUDA_C_64F   |  CUDA_C_64F   |  CUDA_C_64F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_R_32F   |  CUDA_R_32F   |  CUDA_R_16F   |  CUDA_R_32F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_R_64F   |  CUDA_R_64F   |  CUDA_R_32F   |  CUDA_R_64F   |
 * +---------------+---------------+---------------+---------------+
 * |  CUDA_C_64F   |  CUDA_C_64F   |  CUDA_C_32F   |  CUDA_C_64F   |
 * +---------------+---------------+---------------+---------------+
 * \endverbatim
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] alpha Scaling factor for A (see equation above) of the type typeScalar. Pointer to the host memory. If alpha is zero, A is not read and the corresponding unary operator is not applied.
 * \param[in] A Multi-mode tensor of type typeA with nmodeA modes. Pointer to the GPU-accessible memory.
 * \param[in] descA A descriptor that holds the information about the data type, modes, and strides of A.
 * \param[in] modeA Array (in host memory) of size descA->numModes that holds the names of the modes of A (e.g., if A_{a,b,c} => modeA = {'a','b','c'}). The modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
 * \param[in] beta Scaling factor for B (see equation above) of the type typeScalar. Pointer to the host memory. If beta is zero, B is not read and the corresponding unary operator is not applied.
 * \param[in] B Multi-mode tensor of type typeB with nmodeB many modes. Pointer to the GPU-accessible memory.
 * \param[in] descB The B descriptor that holds information about the data type, modes, and strides of B.
 * \param[in] modeB Array (in host memory) of size descB->numModes that holds the names of the modes of B. modeB[i] corresponds to extent[i] and stride[i] of the cutensorInitTensorDescriptor
 * \param[in] gamma Scaling factor for C (see equation above) of type typeScalar. Pointer to the host memory. If gamma is zero, C is not read and the corresponding unary operator is not applied.
 * \param[in] C Multi-mode tensor of type typeC with nmodeC many modes. Pointer to the GPU-accessible memory.
 * \param[in] descC The C descriptor that holds information about the data type, modes, and strides of C.
 * \param[in] modeC Array (in host memory) of size descC->numModes that holds the names of the modes of C. The modeC[i] corresponds to extent[i] and stride[i] of the cutensorInitTensorDescriptor.
 * \param[out] D Multi-mode output tensor of type typeC with nmodeC modes that are ordered according to modeD. Pointer to the GPU-accessible memory. Notice that D may alias any input tensor if they share the same memory layout (i.e., same tensor descriptor).
 * \param[in] descD The D descriptor that holds information about the data type, modes, and strides of D. Notice that we currently request descD and descC to be identical.
 * \param[in] modeD Array (in host memory) of size descD->numModes that holds the names of the modes of D. The modeD[i] corresponds to extent[i] and stride[i] of the cutensorInitTensorDescriptor.
 * \param[in] opAB Element-wise binary operator (see \f$\Phi_{AB}\f$ above).
 * \param[in] opABC Element-wise binary operator (see \f$\Phi_{ABC}\f$ above).
 * \param[in] typeScalar Denotes the data type for the scalars alpha, beta, and gamma. Moreover, typeScalar determines the data type that is used throughout the computation.
 * \param[in] stream The cuda stream.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 * \retval CUTENSOR_STATUS_ARCH_MISMATCH if the device is either not ready, or the target architecture is not supported.
 * \remarks calls asynchronous functions, no reentrant, and thread-safe
 *
 */
cutensorStatus_t cutensorElementwiseTrinary(const cutensorHandle_t* handle,
                 const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
                 const void* beta,  const void* B, const cutensorTensorDescriptor_t* descB, const int32_t modeB[],
                 const void* gamma, const void* C, const cutensorTensorDescriptor_t* descC, const int32_t modeC[],
                                          void* D, const cutensorTensorDescriptor_t* descD, const int32_t modeD[],
                 cutensorOperator_t opAB, cutensorOperator_t opABC, cudaDataType_t typeScalar, const cudaStream_t stream);

/**
 * \brief Element-wise tensor operation for two input tensors
 *
 * \details This function performs a element-wise tensor operation of the form:
 * \f[ D_{\Pi^C(i_0,i_1,...,i_n)} = \Phi_{AC}(\alpha \Psi_A(A_{\Pi^A(i_0,i_1,...,i_n)}), \gamma \Psi_C(C_{\Pi^C(i_0,i_1,...,i_n)})) \f]
 *
 * See cutensorElementwiseTrinary() for details.
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] alpha Scaling factor for A (see equation above) of the type typeScalar. Pointer to the host memory. If alpha is zero, A is not read and the corresponding unary operator is not applied.
 * \param[in] A Multi-mode tensor of type typeA with nmodeA modes. Pointer to the GPU-accessible memory.
 * \param[in] descA A descriptor that holds the information about the data type, modes, and strides of A.
 * \param[in] modeA Array (in host memory) of size descA->numModes that holds the names of the modes of A (e.g., if A_{a,b,c} => modeA = {'a','b','c'}). The modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
 * \param[in] gamma Scaling factor for C (see equation above) of type typeScalar. Pointer to the host memory. If gamma is zero, C is not read and the corresponding unary operator is not applied.
 * \param[in] C Multi-mode tensor of type typeC with nmodeC many modes. Pointer to the GPU-accessible memory.
 * \param[in] descC The C descriptor that holds information about the data type, modes, and strides of C.
 * \param[in] modeC Array (in host memory) of size descC->numModes that holds the names of the modes of C. The modeC[i] corresponds to extent[i] and stride[i] of the cutensorInitTensorDescriptor.
 * \param[out] D Multi-mode output tensor of type typeC with nmodeC modes that are ordered according to modeD. Pointer to the GPU-accessible memory. Notice that D may alias any input tensor if they share the same memory layout (i.e., same tensor descriptor).
 * \param[in] descD The D descriptor that holds information about the data type, modes, and strides of D. Notice that we currently request descD and descC to be identical.
 * \param[in] modeD Array (in host memory) of size descD->numModes that holds the names of the modes of D. The modeD[i] corresponds to extent[i] and stride[i] of the cutensorInitTensorDescriptor.
 * \param[in] opAC Element-wise binary operator (see \f$\Phi_{AC}\f$ above).
 * \param[in] typeScalar Scalar type for the intermediate computation.
 * \param[in] stream The cuda stream.
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
 * \retval CUTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully without error
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \remarks calls asynchronous functions, no reentrant, and thread-safe
 */
cutensorStatus_t cutensorElementwiseBinary(const cutensorHandle_t* handle,
                 const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
                 const void* gamma, const void* C, const cutensorTensorDescriptor_t* descC, const int32_t modeC[],
                                          void* D, const cutensorTensorDescriptor_t* descD, const int32_t modeD[],
                 cutensorOperator_t opAC, cudaDataType_t typeScalar, cudaStream_t stream);

/**
 * \brief Tensor permutation
 * \details This function performs an element-wise tensor operation of the form:
 * \f[ B_{\Pi^B(i_0,i_1,...,i_n)} = \alpha \Psi(A_{\Pi^A(i_0,i_1,...,i_n)}) \f]
 *
 * Consequently, this function performs an out-of-place tensor permutation and is a specialization of cutensorElementwise.
 *
 * Where
 *    - A and B are multi-mode tensors (of arbitrary data types),
 *    - \f$\Pi^A, \Pi^B\f$ are permutation operators that permute the modes of A, B respectively,
 *    - \f$\Psi\f$ is an unary element-wise operators (e.g., IDENTITY, SQR, CONJUGATE), and
 *    - \f$\Psi\f$ is specified in the tensor descriptor descA.
 *
 * Broadcasting (of a mode) can be achieved by simply omitting that mode from the respective tensor.
 *
 * Modes may appear in any order. The only <b>restrictions</b> are:
 *    - modes that appear in A _must_ also appear in the output tensor.
 *    - each mode may appear in each tensor at most once.
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] alpha Scaling factor for A (see equation above) of the type typeScalar. Pointer to the host memory. If alpha is zero, A is not read and the corresponding unary operator is not applied.
 * \param[in] A Multi-mode tensor of type typeA with nmodeA modes. Pointer to the GPU-accessible memory.
 * \param[in] descA A descriptor that holds information about the data type, modes, and strides of A.
 * \param[in] modeA Array of size descA->numModes that holds the names of the modes of A (e.g., if A_{a,b,c} => modeA = {'a','b','c'})
 * \param[in,out] B Multi-mode tensor of type typeB with nmodeB modes. Pointer to the GPU-accessible memory.
 * \param[in] descB A descriptor that holds information about the data type, modes, and strides of B.
 * \param[in] modeB Array of size descB->numModes that holds the names of the modes of B
 * \param[in] typeScalar data type of alpha
 * \param[in] stream The CUDA stream.
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
 * \retval CUTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully without error
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \remarks calls asynchronous functions, no reentrant, and thread-safe
 */
cutensorStatus_t cutensorPermutation(const cutensorHandle_t* handle,
                 const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
                                          void* B, const cutensorTensorDescriptor_t* descB, const int32_t modeB[],
                 const cudaDataType_t typeScalar, const cudaStream_t stream );

/**
 * \brief Describes the tensor contraction problem of the form: \f[ D = \alpha \mathcal{A}  \mathcal{B} + \beta \mathcal{C} \f]
 *
 * \details \f[ \mathcal{D}_{{modes}_\mathcal{D}} \gets \alpha \mathcal{A}_{{modes}_\mathcal{A}} B_{{modes}_\mathcal{B}} + \beta \mathcal{C}_{{modes}_\mathcal{C}} \f].
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[out] desc This opaque struct gets filled with the information that encodes
 * the tensor contraction problem.
 * \param[in] descA A descriptor that holds the information about the data type, modes and strides of A.
 * \param[in] modeA Array with 'nmodeA' entries that represent the modes of A. The modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
 * \param[in] alignmentRequirementA Alignment that cuTENSOR may require for A's pointer (in bytes); you
 * can use the helper function \ref cutensorGetAlignmentRequirement to determine the best value for a given pointer.
 * \param[in] descB The B descriptor that holds information about the data type, modes, and strides of B.
 * \param[in] modeB Array with 'nmodeB' entries that represent the modes of B. The modeB[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
 * \param[in] alignmentRequirementB Alignment that cuTENSOR may require for B's pointer (in bytes); you
 * can use the helper function \ref cutensorGetAlignmentRequirement to determine the best value for a given pointer.
 * \param[in] modeC Array with 'nmodeC' entries that represent the modes of C. The modeC[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
 * \param[in] descC The C descriptor that holds information about the data type, modes, and strides of C.
 * \param[in] alignmentRequirementC Alignment that cuTENSOR may require for C's pointer (in bytes); you
 * can use the helper function \ref cutensorGetAlignmentRequirement to determine the best value for a given pointer.
 * \param[in] modeD Array with 'nmodeD' entries that represent the modes of D (must be identical to modeC for now). The modeD[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
 * \param[in] descD The D descriptor that holds information about the data type, modes, and strides of D (must be identical to descC for now).
 * \param[in] alignmentRequirementD Alignment that cuTENSOR may require for D's pointer (in bytes); you
 * can use the helper function \ref cutensorGetAlignmentRequirement to determine the best value for a given pointer.
 * \param[in] typeCompute Datatype of for the intermediate computation of typeCompute T = A * B.
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED if the combination of data types or operations is not supported
 * \retval CUTENSOR_STATUS_INVALID_VALUE if tensor dimensions or modes have an illegal value
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully without error
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 */
cutensorStatus_t cutensorInitContractionDescriptor(const cutensorHandle_t* handle,
                                                   cutensorContractionDescriptor_t* desc,
                                                   const cutensorTensorDescriptor_t* descA, const int32_t modeA[], const uint32_t alignmentRequirementA,
                                                   const cutensorTensorDescriptor_t* descB, const int32_t modeB[], const uint32_t alignmentRequirementB,
                                                   const cutensorTensorDescriptor_t* descC, const int32_t modeC[], const uint32_t alignmentRequirementC,
                                                   const cutensorTensorDescriptor_t* descD, const int32_t modeD[], const uint32_t alignmentRequirementD,
                                                   cutensorComputeType_t typeCompute);

/**
 * \brief Sett attribute for cutensorDescriptor
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in,out] desc Contraction descriptor that will be modified.
 * \param[in] attr Specifies the attribute that will be set.
 * \param[in] buf This buffer (of size sizeInBytes) determines the value to which attr
 * will be set.
 * \param[in] sizeInBytes Size of buf (in bytes).
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 */
cutensorStatus_t cutensorContractionDescriptorSetAttribute(
        const cutensorHandle_t* handle,
        cutensorContractionDescriptor_t* desc,
        cutensorContractionDescriptorAttributes_t attr,
        const void *buf,
        size_t sizeInBytes);

/**
 * \brief Limits the search space of viable candidates (a.k.a. algorithms)
 *
 * \details This function gives the user finer control over the candidates that the subsequent call to \ref cutensorInitContractionPlan
 * is allowed to evaluate.
 * 
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[out] find
 * \param[in] algo Allows users to select a specific algorithm. CUTENSOR_ALGO_DEFAULT lets the heuristic choose the algorithm. Any value >= 0 selects a specific GEMM-like algorithm
 *                 and deactivates the heuristic. If a specified algorithm is not supported CUTENSOR_STATUS_NOT_SUPPORTED is returned. See \ref cutensorAlgo_t for additional choices.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 */
cutensorStatus_t cutensorInitContractionFind(const cutensorHandle_t* handle,
                                             cutensorContractionFind_t* find,
                                             const cutensorAlgo_t algo);

/**
 * \brief Set attribute for cutensorContractionFind
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in,out] find This opaque struct restricts the search space of viable candidates.
 * \param[in] attr Specifies the attribute that will be set.
 * \param[in] buf This buffer (of size sizeInBytes) determines the value to which attr
 * will be set.
 * \param[in] sizeInBytes Size of buf (in bytes).
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 */
cutensorStatus_t cutensorContractionFindSetAttribute(
        const cutensorHandle_t* handle,
        cutensorContractionFind_t* find,
        cutensorContractionFindAttributes_t attr,
        const void *buf,
        size_t sizeInBytes);

/**
 * \brief Determines the required workspaceSize for a given tensor contraction (see \ref cutensorContraction)
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] desc This opaque struct encodes the tensor contraction problem.
 * \param[in] find This opaque struct restricts the search space of viable candidates.
 * \param[in] pref This parameter influences the size of the workspace; see \ref cutensorWorksizePreference_t for details.
 * \param[out] workspaceSize The workspace size (in bytes) that is required for the given tensor contraction.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 */
cutensorStatus_t cutensorContractionGetWorkspace(const cutensorHandle_t* handle,
                                                 const cutensorContractionDescriptor_t* desc,
                                                 const cutensorContractionFind_t* find,
                                                 const cutensorWorksizePreference_t pref,
                                                 uint64_t *workspaceSize);

/**
 * \brief Initializes the contraction plan for a given tensor contraction problem
 *
 * \details This function applies cuTENSOR's heuristic to select a candidate for a
 * given tensor contraction problem (encoded by desc). The resulting plan can be reused
 * multiple times as long as the tensor contraction problem remains the same.
 *
 * The plan is created for the active CUDA device.
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[out] plan Opaque handle holding the contraction execution plan (i.e., the
 * candidate that will be executed as well as all it's runtime parameters for the given
 * tensor contraction problem).
 * \param[in] desc This opaque struct encodes the given tensor contraction problem.
 * \param[in] find This opaque struct is used to restrict the search space of viable candidates.
 * \param[in] workspaceSize Available workspace size (in bytes).
 *
 * \retval CUTENSOR_STATUS_SUCCESS If a viable candidate has been found.
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED If no viable candidate could be found.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE if The provided workspace was insufficient.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 */
cutensorStatus_t cutensorInitContractionPlan(const cutensorHandle_t* handle,
                                             cutensorContractionPlan_t* plan,
                                             const cutensorContractionDescriptor_t* desc,
                                             const cutensorContractionFind_t* find,
                                             const uint64_t workspaceSize);

/**
 * \brief This routine computes the tensor contraction \f[ D = alpha * A * B + beta * C \f]
 *
 * \details \f[ \mathcal{D}_{{modes}_\mathcal{D}} \gets \alpha * \mathcal{A}_{{modes}_\mathcal{A}} B_{{modes}_\mathcal{B}} + \beta \mathcal{C}_{{modes}_\mathcal{C}} \f]
 *
 * The currently active CUDA device must match the CUDA device that was active at the time at which the plan was created.
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |     typeA     |     typeB     |     typeC     |        typeCompute      | Tensor Core |
 * +===============+===============+===============+=========================+=============+
 * |  CUDA_R_16F   |  CUDA_R_16F   |  CUDA_R_16F   |  CUTENSOR_COMPUTE_32F   | Volta+      |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_R_16BF  |  CUDA_R_16BF  |  CUDA_R_16BF  |  CUTENSOR_COMPUTE_32F   | Ampere+     |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_R_32F   |  CUDA_R_32F   |  CUDA_R_32F   |  CUTENSOR_COMPUTE_32F   | No          |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_R_32F   |  CUDA_R_32F   |  CUDA_R_32F   |  CUTENSOR_COMPUTE_TF32  | Ampere+     |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_R_32F   |  CUDA_R_32F   |  CUDA_R_32F   |  CUTENSOR_COMPUTE_16BF  | Ampere+     |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_R_32F   |  CUDA_R_32F   |  CUDA_R_32F   |  CUTENSOR_COMPUTE_16F   | Volta+      |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_R_64F   |  CUDA_R_64F   |  CUDA_R_64F   |  CUTENSOR_COMPUTE_64F   | Ampere+     |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_R_64F   |  CUDA_R_64F   |  CUDA_R_64F   |  CUTENSOR_COMPUTE_32F   | No          |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_C_32F   |  CUDA_C_32F   |  CUDA_C_32F   |  CUTENSOR_COMPUTE_32F   | No          |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_C_32F   |  CUDA_C_32F   |  CUDA_C_32F   |  CUTENSOR_COMPUTE_TF32  | Ampere+     |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_C_64F   |  CUDA_C_64F   |  CUDA_C_64F   |  CUTENSOR_COMPUTE_64F   | Ampere+     |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_C_64F   |  CUDA_C_64F   |  CUDA_C_64F   |  CUTENSOR_COMPUTE_32F   | No          |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_R_64F   |  CUDA_C_64F   |  CUDA_C_64F   |  CUTENSOR_COMPUTE_64F   | No          |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * |  CUDA_C_64F   |  CUDA_R_64F   |  CUDA_C_64F   |  CUTENSOR_COMPUTE_64F   | No          |
 * +---------------+---------------+---------------+-------------------------+-------------+
 * \endverbatim
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] plan Opaque handle holding the contraction execution plan.
 * \param[in] alpha Scaling for A*B. Its data type is determined by 'typeCompute'. Pointer to the host memory.
 * \param[in] A Pointer to the data corresponding to A in device memory. Pointer to the GPU-accessible memory.
 * \param[in] B Pointer to the data corresponding to B. Pointer to the GPU-accessible memory.
 * \param[in] beta Scaling for C. Its data type is determined by 'typeCompute'. Pointer to the host memory.
 * \param[in] C Pointer to the data corresponding to C. Pointer to the GPU-accessible memory.
 * \param[out] D Pointer to the data corresponding to D. Pointer to the GPU-accessible memory.
 * \param[out] workspace Optional parameter that may be NULL. This pointer provides additional workspace, in device memory, to the library for additional optimizations; the workspace must be aligned to 128 bytes.
 * \param[in] workspaceSize Size of the workspace array in bytes; please refer to cutensorContractionGetWorkspace() to query the required workspace. While cutensorContraction() does not strictly require a workspace for the reduction, it is still recommended to provided some small workspace (e.g., 128 MB).
 * \param[in] stream The CUDA stream in which all the computation is performed.
 *
 * \par[Example]
 * See https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR/contraction.cu for a concrete example.
 *
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED if operation is not supported.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_ARCH_MISMATCH if the plan was created for a different device than the currently active device.
 * \retval CUTENSOR_STATUS_INSUFFICIENT_DRIVER if the driver is insufficient.
 * \retval CUTENSOR_STATUS_CUDA_ERROR if some unknown CUDA error has occurred (e.g., out of memory).
 */
cutensorStatus_t cutensorContraction(const cutensorHandle_t* handle, 
                                     const cutensorContractionPlan_t* plan,
                                     const void* alpha, const void* A, const void* B,
                                     const void* beta,  const void* C,       void* D,
                                     void *workspace, uint64_t workspaceSize, cudaStream_t stream);

/**
 * \brief This routine returns the maximum number of algorithms available to compute tensor contractions
 * \param[out] maxNumAlgos This value will hold the maximum number of algorithms available for cutensorContraction().
 *                      You can use the returned integer for auto-tuning purposes (i.e., iterate over all algorithms up to the returned value).
 *
 * \par[NOTE] Not all algorithms might be applicable to your specific problem. cutensorContraction() will return CUTENSOR_STATUS_NOT_SUPPORTED if an algorithm is not applicable.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 */
cutensorStatus_t cutensorContractionMaxAlgos(int32_t* maxNumAlgos);

/**
 * \brief Implements a tensor reduction of the form \f[ D = alpha * opReduce(opA(A)) + beta * opC(C) \f]
 *
 * \details
 * For example this function enables users to reduce an entire tensor to a scalar: C[] = alpha * A[i,j,k];
 *
 * This function is also able to perform partial reductions; for instance: C[i,j] = alpha * A[k,j,i]; in this case only elements along the k-mode are contracted.
 *
 * The binary opReduce operator provides extra control over what kind of a reduction
 * ought to be perfromed. For instance, opReduce == CUTENSOR_OP_ADD reduces element of A
 * via a summation while CUTENSOR_OP_MAX would find the largest element in A.
 *
 * Supported data-type combinations are:
 *
 * \verbatim embed:rst:leading-asterisk
 * +---------------+---------------+---------------+-------------------------+
 * |     typeA     |     typeB     |     typeC     |       typeCompute       |
 * +===============+===============+===============+=========================+
 * | `CUDA_R_16F`  | `CUDA_R_16F`  | `CUDA_R_16F`  | `CUTENSOR_COMPUTE_16F`  |
 * +---------------+---------------+---------------+-------------------------+
 * | `CUDA_R_16F`  | `CUDA_R_16F`  | `CUDA_R_16F`  | `CUTENSOR_COMPUTE_32F`  |
 * +---------------+---------------+---------------+-------------------------+
 * | `CUDA_R_16BF` | `CUDA_R_16BF` | `CUDA_R_16BF` | `CUTENSOR_COMPUTE_16BF` |
 * +---------------+---------------+---------------+-------------------------+
 * | `CUDA_R_16BF` | `CUDA_R_16BF` | `CUDA_R_16BF` | `CUTENSOR_COMPUTE_32F`  |
 * +---------------+---------------+---------------+-------------------------+
 * | `CUDA_R_32F`  | `CUDA_R_32F`  | `CUDA_R_32F`  | `CUTENSOR_COMPUTE_32F`  |
 * +---------------+---------------+---------------+-------------------------+
 * | `CUDA_R_64F`  | `CUDA_R_64F`  | `CUDA_R_64F`  | `CUTENSOR_COMPUTE_64F`  |
 * +---------------+---------------+---------------+-------------------------+
 * | `CUDA_C_32F`  | `CUDA_C_32F`  | `CUDA_C_32F`  | `CUTENSOR_COMPUTE_32F`  |
 * +---------------+---------------+---------------+-------------------------+
 * | `CUDA_C_64F`  | `CUDA_C_64F`  | `CUDA_C_64F`  | `CUTENSOR_COMPUTE_64F`  |
 * +---------------+---------------+---------------+-------------------------+
 * \endverbatim
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] alpha Scaling for A; its data type is determined by 'typeCompute'. Pointer to the host memory.
 * \param[in] A Pointer to the data corresponding to A in device memory. Pointer to the GPU-accessible memory.
 * \param[in] descA A descriptor that holds the information about the data type, modes and strides of A.
 * \param[in] modeA Array with 'nmodeA' entries that represent the modes of A. modeA[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor. Modes that only appear in modeA but not in modeC are reduced (contracted).
 * \param[in] beta Scaling for C; its data type is determined by 'typeCompute'. Pointer to the host memory.
 * \param[in] C Pointer to the data corresponding to C in device memory. Pointer to the GPU-accessible memory.
 * \param[in] descC A descriptor that holds the information about the data type, modes and strides of C.
 * \param[in] modeC Array with 'nmodeC' entries that represent the modes of C. modeC[i] corresponds to extent[i] and stride[i] w.r.t. the arguments provided to cutensorInitTensorDescriptor.
 * \param[out] D Pointer to the data corresponding to C in device memory. Pointer to the GPU-accessible memory.
 * \param[in] descD Must be identical to descC for now.
 * \param[in] modeD Must be identical to modeC for now.
 * \param[in] opReduce binary operator used to reduce elements of A.
 * \param[in] typeCompute All arithmetic is performed using this data type (i.e., it affects the accuracy and performance).
 * \param[out] workspace Scratchpad (device) memory; the workspace must be aligned to 128 bytes.
 * \param[in] workspaceSize Please use cutensorReductionGetWorkspace() to query the required workspace.
 *            While lower values, including zero, are valid, they may lead to grossly suboptimal performance.
 * \param[in] stream The CUDA stream in which all the computation is performed.
 *
 * \retval CUTENSOR_STATUS_NOT_SUPPORTED if operation is not supported.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 *
 */
cutensorStatus_t cutensorReduction(const cutensorHandle_t* handle, 
        const void* alpha, const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
        const void* beta,  const void* C, const cutensorTensorDescriptor_t* descC, const int32_t modeC[],
                                 void* D, const cutensorTensorDescriptor_t* descD, const int32_t modeD[],
       cutensorOperator_t opReduce, cutensorComputeType_t typeCompute, void *workspace, uint64_t workspaceSize,
       cudaStream_t stream);



/**
 * \brief Determines the required workspaceSize for a given tensor reduction (see \ref cutensorReduction)
 *
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] A same as in cutensorReduction
 * \param[in] descA same as in cutensorReduction
 * \param[in] modeA same as in cutensorReduction
 * \param[in] C same as in cutensorReduction
 * \param[in] descC same as in cutensorReduction
 * \param[in] modeC same as in cutensorReduction
 * \param[in] D same as in cutensorReduction
 * \param[in] descD same as in cutensorReduction
 * \param[in] modeD same as in cutensorReduction
 * \param[in] opReduce same as in cutensorReduction
 * \param[in] typeCompute same as in cutensorReduction
 * \param[out] workspaceSize The workspace size (in bytes) that is required for the given tensor reduction.
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 * \retval CUTENSOR_STATUS_INVALID_VALUE if some input data is invalid (this typically indicates an user error).
 */
cutensorStatus_t cutensorReductionGetWorkspace(const cutensorHandle_t* handle, 
        const void* A, const cutensorTensorDescriptor_t* descA, const int32_t modeA[],
        const void* C, const cutensorTensorDescriptor_t* descC, const int32_t modeC[],
        const void* D, const cutensorTensorDescriptor_t* descD, const int32_t modeD[],
        cutensorOperator_t opReduce, cutensorComputeType_t typeCompute, uint64_t *workspaceSize);

/**
 * \brief Computes the minimal alignment requirement for a given pointer and descriptor
 * \param[in] handle Opaque handle holding cuTENSOR's library context.
 * \param[in] ptr Raw pointer to the data of the respective tensor.
 * \param[in] desc Tensor descriptor for ptr.
 * \param[out] alignmentRequirement Largest alignment requirement that ptr can fulfill (in bytes).
 * \retval CUTENSOR_STATUS_SUCCESS The operation completed successfully.
 * \retval CUTENSOR_STATUS_NOT_INITIALIZED if the handle is not initialized.
 */
cutensorStatus_t cutensorGetAlignmentRequirement(const cutensorHandle_t* handle,
                                                 const void* ptr,
                                                 const cutensorTensorDescriptor_t *desc,
                                                 uint32_t *alignmentRequirement);

/**
 * \brief Returns the description string for an error code
 * \param[in] error Error code to convert to string.
 * \returns the error string
 * \remarks non-blocking, no reentrant, and thread-safe
 */
const char *cutensorGetErrorString(const cutensorStatus_t error);

/**
 * \brief Returns Version number of the CUTENSOR library
 */
size_t cutensorGetVersion();

/**
 * \brief Returns version number of the CUDA runtime that cuTENSOR was compiled against
 * \details Can be compared against the CUDA runtime version from cudaRuntimeGetVersion().
 */
size_t cutensorGetCudartVersion();

/**
 * \brief This function sets the logging callback routine.
 * \param[in] callback Pointer to a callback function. Check cutensorLoggerCallback_t.
 */
cutensorStatus_t cutensorLoggerSetCallback(cutensorLoggerCallback_t callback);

/**
 * \brief This function sets the logging output file.
 * \param[in] file An open file with write permission.
 */
cutensorStatus_t cutensorLoggerSetFile(FILE* file);

/**
 * \brief This function opens a logging output file in the given path.
 * \param[in] logFile Path to the logging output file.
 */
cutensorStatus_t cutensorLoggerOpenFile(const char* logFile);

/**
 * \brief This function sets the value of the logging level.
 * \param[in] level Log level, should be one of the following:
 *                  0.  Off
 *                  1.  Errors
 *                  2.  Performance Trace
 *                  3.  Performance Hints
 *                  4.  Heuristics Trace
 *                  5.  API Trace
 */
cutensorStatus_t cutensorLoggerSetLevel(int32_t level);

/**
 * \brief This function sets the value of the log mask.
 * \param[in] mask Log mask, the bitwise OR of the following:
 *                 0.  Off
 *                 1.  Errors
 *                 2.  Performance Trace
 *                 4.  Performance Hints
 *                 8.  Heuristics Trace
 *                 16. API Trace
 *
 */
cutensorStatus_t cutensorLoggerSetMask(int32_t mask);

/**
 * \brief This function disables logging for the entire run.
 */
cutensorStatus_t cutensorLoggerForceDisable();

#if defined(__cplusplus)
}
#endif /* __cplusplus */
