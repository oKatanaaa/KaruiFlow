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
 * @brief This file defines the types provided by the cuTENSOR library.
 */
#pragma once

#include <stdint.h>

/**
 * \brief This enum captures all unary and binary element-wise operations supported by the cuTENSOR library.
 * \ingroup runtimeDataStructurePLC3
 */
typedef enum 
{
    /* Unary */
    CUTENSOR_OP_IDENTITY = 1,          ///< Identity operator (i.e., elements are not changed)
    CUTENSOR_OP_SQRT = 2,              ///< Square root
    CUTENSOR_OP_RELU = 8,              ///< Rectified linear unit
    CUTENSOR_OP_CONJ = 9,              ///< Complex conjugate
    CUTENSOR_OP_RCP = 10,              ///< Reciprocal
    CUTENSOR_OP_SIGMOID = 11,          ///< y=1/(1+exp(-x))
    CUTENSOR_OP_TANH = 12,             ///< y=tanh(x)
    CUTENSOR_OP_EXP = 22,              ///< Exponentiation.
    CUTENSOR_OP_LOG = 23,              ///< Log (base e).
    CUTENSOR_OP_ABS = 24,              ///< Absolute value.
    CUTENSOR_OP_NEG = 25,              ///< Negation.
    CUTENSOR_OP_SIN = 26,              ///< Sine.
    CUTENSOR_OP_COS = 27,              ///< Cosine.
    CUTENSOR_OP_TAN = 28,              ///< Tangent.
    CUTENSOR_OP_SINH = 29,             ///< Hyperbolic sine.
    CUTENSOR_OP_COSH = 30,             ///< Hyperbolic cosine.
    CUTENSOR_OP_ASIN = 31,             ///< Inverse sine.
    CUTENSOR_OP_ACOS = 32,             ///< Inverse cosine.
    CUTENSOR_OP_ATAN = 33,             ///< Inverse tangent.
    CUTENSOR_OP_ASINH = 34,            ///< Inverse hyperbolic sine.
    CUTENSOR_OP_ACOSH = 35,            ///< Inverse hyperbolic cosine.
    CUTENSOR_OP_ATANH = 36,            ///< Inverse hyperbolic tangent.
    CUTENSOR_OP_CEIL = 37,             ///< Ceiling.
    CUTENSOR_OP_FLOOR = 38,            ///< Floor.
    /* Binary */
    CUTENSOR_OP_ADD = 3,               ///< Addition of two elements
    CUTENSOR_OP_MUL = 5,               ///< Multiplication of two elements
    CUTENSOR_OP_MAX = 6,               ///< Maximum of two elements
    CUTENSOR_OP_MIN = 7,               ///< Minimum of two elements

    CUTENSOR_OP_UNKNOWN = 126, ///< reserved for internal use only

} cutensorOperator_t;

/**
 * \brief cuTENSOR status type returns
 *
 * \details The type is used for function status returns. All cuTENSOR library functions return their status, which can have the following values.
 * \ingroup runtimeDataStructurePLC3
 */
typedef enum 
{
    /** The operation completed successfully.*/
    CUTENSOR_STATUS_SUCCESS                = 0,
    /** The cuTENSOR library was not initialized.*/
    CUTENSOR_STATUS_NOT_INITIALIZED        = 1,
    /** Resource allocation failed inside the cuTENSOR library.*/
    CUTENSOR_STATUS_ALLOC_FAILED           = 3,
    /** An unsupported value or parameter was passed to the function (indicates an user error).*/
    CUTENSOR_STATUS_INVALID_VALUE          = 7,
    /** Indicates that the device is either not ready, or the target architecture is not supported.*/
    CUTENSOR_STATUS_ARCH_MISMATCH          = 8,
    /** An access to GPU memory space failed, which is usually caused by a failure to bind a texture.*/
    CUTENSOR_STATUS_MAPPING_ERROR          = 11,
    /** The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.*/
    CUTENSOR_STATUS_EXECUTION_FAILED       = 13,
    /** An internal cuTENSOR error has occurred.*/
    CUTENSOR_STATUS_INTERNAL_ERROR         = 14,
    /** The requested operation is not supported.*/
    CUTENSOR_STATUS_NOT_SUPPORTED          = 15,
    /** The functionality requested requires some license and an error was detected when trying to check the current licensing.*/
    CUTENSOR_STATUS_LICENSE_ERROR          = 16,
    /** A call to CUBLAS did not succeed.*/
    CUTENSOR_STATUS_CUBLAS_ERROR           = 17,
    /** Some unknown CUDA error has occurred.*/
    CUTENSOR_STATUS_CUDA_ERROR             = 18,
    /** The provided workspace was insufficient.*/
    CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE = 19,
    /** Indicates that the driver version is insufficient.*/
    CUTENSOR_STATUS_INSUFFICIENT_DRIVER    = 20,
    /** Indicates an error related to file I/O.*/
    CUTENSOR_STATUS_IO_ERROR               = 21,
} cutensorStatus_t;

/**
 * \brief Allows users to specify the algorithm to be used for performing the
 * tensor contraction.
 *
 * \details This enum gives users finer control over which algorithm should be executed by
 * cutensorContraction(); values >= 0 correspond to certain sub-algorithms of GETT.
 */
typedef enum
{
    CUTENSOR_ALGO_DEFAULT_PATIENT   = -6, ///< Uses the more accurate but also more time-consuming performance model
    CUTENSOR_ALGO_GETT              = -4, ///< Choose the GETT algorithm
    CUTENSOR_ALGO_TGETT             = -3, ///< Transpose (A or B) + GETT
    CUTENSOR_ALGO_TTGT              = -2, ///< Transpose-Transpose-GEMM-Transpose (requires additional memory)
    CUTENSOR_ALGO_DEFAULT           = -1, ///< Lets the internal heuristic choose
} cutensorAlgo_t;

/**
 * \brief This enum gives users finer control over the suggested workspace
 *
 * \details This enum gives users finer control over the amount of workspace that is
 * suggested by cutensorContractionGetWorkspace
 */
typedef enum
{
    CUTENSOR_WORKSPACE_MIN = 1,         ///< At least one algorithm will be available
    CUTENSOR_WORKSPACE_RECOMMENDED = 2, ///< The most suitable algorithm will be available
    CUTENSOR_WORKSPACE_MAX = 3,         ///< All algorithms will be available
} cutensorWorksizePreference_t;

/**
 * \brief Encodes cuTENSOR's compute type (see "User Guide - Accuracy Guarantees" for details).
 */
typedef enum
{
    CUTENSOR_COMPUTE_16F  = (1U<< 0U),  ///< floating-point: 5-bit exponent and 10-bit mantissa (aka half)
    CUTENSOR_COMPUTE_16BF = (1U<< 10U),  ///< floating-point: 8-bit exponent and 7-bit mantissa (aka bfloat)
    CUTENSOR_COMPUTE_TF32 = (1U<< 12U),  ///< floating-point: 8-bit exponent and 10-bit mantissa (aka tensor-float-32)
    CUTENSOR_COMPUTE_32F  = (1U<< 2U),  ///< floating-point: 8-bit exponent and 23-bit mantissa (aka float)
    CUTENSOR_COMPUTE_64F  = (1U<< 4U),  ///< floating-point: 11-bit exponent and 52-bit mantissa (aka double)
    CUTENSOR_COMPUTE_8U   = (1U<< 6U),  ///< 8-bit unsigned integer
    CUTENSOR_COMPUTE_8I   = (1U<< 8U),  ///< 8-bit signed integer
    CUTENSOR_COMPUTE_32U  = (1U<< 7U),  ///< 32-bit unsigned integer
    CUTENSOR_COMPUTE_32I  = (1U<< 9U),  ///< 32-bit signed integer
   
    /* All compute types below this line will be deprecated in the near future. */
    CUTENSOR_R_MIN_16F  = (1U<< 0U),  ///< DEPRECATED (real as a half), please use CUTENSOR_COMPUTE_16F instead
    CUTENSOR_C_MIN_16F  = (1U<< 1U),  ///< DEPRECATED (complex as a half), please use CUTENSOR_COMPUTE_16F instead
    CUTENSOR_R_MIN_32F  = (1U<< 2U),  ///< DEPRECATED (real as a float), please use CUTENSOR_COMPUTE_32F instead
    CUTENSOR_C_MIN_32F  = (1U<< 3U),  ///< DEPRECATED (complex as a float), please use CUTENSOR_COMPUTE_32F instead
    CUTENSOR_R_MIN_64F  = (1U<< 4U),  ///< DEPRECATED (real as a double), please use CUTENSOR_COMPUTE_64F instead
    CUTENSOR_C_MIN_64F  = (1U<< 5U),  ///< DEPRECATED (complex as a double), please use CUTENSOR_COMPUTE_64F instead
    CUTENSOR_R_MIN_8U   = (1U<< 6U),  ///< DEPRECATED (real as a uint8), please use CUTENSOR_COMPUTE_8U instead
    CUTENSOR_R_MIN_32U  = (1U<< 7U),  ///< DEPRECATED (real as a uint32), please use CUTENSOR_COMPUTE_32U instead
    CUTENSOR_R_MIN_8I   = (1U<< 8U),  ///< DEPRECATED (real as a int8), please use CUTENSOR_COMPUTE_8I instead
    CUTENSOR_R_MIN_32I  = (1U<< 9U),  ///< DEPRECATED (real as a int32), please use CUTENSOR_COMPUTE_32I instead
    CUTENSOR_R_MIN_16BF = (1U<<10U),  ///< DEPRECATED (real as a bfloat16), please use CUTENSOR_COMPUTE_16BF instead
    CUTENSOR_R_MIN_TF32 = (1U<<11U),  ///< DEPRECATED (real as a tensorfloat32), please use CUTENSOR_COMPUTE_TF32 instead
    CUTENSOR_C_MIN_TF32 = (1U<<12U),  ///< DEPRECATED (complex as a tensorfloat32), please use CUTENSOR_COMPUTE_TF32 instead
} cutensorComputeType_t;

/**
 * This enum lists all attributes of a cutensorContractionDescriptor_t that can be modified.
 */
typedef enum
{
    CUTENSOR_CONTRACTION_DESCRIPTOR_TAG ///< uint32_t: enables users to distinguish two identical tensor contractions w.r.t. the sw-managed plan-cache. (default value: 0)
} cutensorContractionDescriptorAttributes_t;

/**
 * This enum lists all attributes of a cutensorContractionFind_t that can be modified.
 */
typedef enum
{
    CUTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE, ///< cutensorAutotuneMode_t: Determines if the corresponding algrithm/kernel for this plan should be cached.
    CUTENSOR_CONTRACTION_FIND_CACHE_MODE, ///< cutensorCacheMode_t: Gives fine control over what is considered a cachehit.
    CUTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT, ///< uint32_t: Only applicable if CUTENSOR_CONTRACTION_FIND_CACHE_MODE is set to CUTENSOR_AUTOTUNE_INCREMENTAL
} cutensorContractionFindAttributes_t;

/**
 * This enum is important w.r.t. cuTENSOR's caching capability of plans.
 */
typedef enum
{
    CUTENSOR_AUTOTUNE_NONE, ///< Indicates no autotuning (default); in this case the cache will help to reduce the plan-creation overhead. In the case of a cachehit: the cached plan will be reused, otherwise the plancache will be neglected.
    CUTENSOR_AUTOTUNE_INCREMENTAL, ///< Indicates an incremental autotuning (i.e., each invocation of corresponding cutensorInitContractionPlan() will create a plan based on a different algorithm/kernel; the maximum number of kernels that will be tested is defined by the CUTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT FindAttributes_t). WARNING: If this autotuning mode is selected, then we cannot guarantee bit-wise identical results (since different algorithms could be executed).
} cutensorAutotuneMode_t;

/**
 * This enum defines what is considered a cache hit.
 */
typedef enum
{
    CUTENSOR_CACHE_MODE_NONE,     ///< Plan will not be cached
    CUTENSOR_CACHE_MODE_PEDANTIC, ///< All parameters of the corresponding descriptor must be identical to the cached plan (default).
} cutensorCacheMode_t;

/**
 * \brief Opaque structure holding cuTENSOR's library context.
 */
typedef struct { int64_t fields[512]; /*!< Data */ } cutensorHandle_t;

/**
 * \brief Opaque data structure that represents a cacheline of the software-managed plan cache
 */
typedef struct { int64_t fields[1408]; /*!< Data */ } cutensorPlanCacheline_t;

/**
 * \brief Opaque data structure that represents the software-managed plan cache. A plan cache must not be used by multiple threads.
 */
typedef struct { int64_t fields[12*1024]; /*!< Data */ } cutensorPlanCache_t;

/**
 * \brief Opaque structure representing a tensor descriptor.
 */
typedef struct { int64_t fields[72]; /*!< Data */ } cutensorTensorDescriptor_t;

/**
 * \brief Opaque structure representing a tensor contraction descriptor.
 */
typedef struct { int64_t fields[288]; /*!< Data */ } cutensorContractionDescriptor_t;

/**
 * \brief Opaque structure representing a plan.
 */
typedef struct { int64_t fields[1408]; /*!< Data */ } cutensorContractionPlan_t;

/**
 * \brief Opaque structure representing a candidate.
 */
typedef struct { int64_t fields[64]; /*!< Data */ } cutensorContractionFind_t;

/**
 * \brief A function pointer type for logging.
 */
typedef void (*cutensorLoggerCallback_t)(
        int32_t logLevel,
        const char* functionName,
        const char* message
);

