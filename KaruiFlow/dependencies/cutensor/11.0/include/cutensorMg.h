/*
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <stdint.h>
#include <cutensor.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

enum cutensorMgHostDevice_t
{
    CUTENSOR_MG_DEVICE_HOST = -1
};

struct cutensorMgHandle_s;
typedef struct cutensorMgHandle_s* cutensorMgHandle_t;

struct cutensorMgTensorDescriptor_s;
typedef struct cutensorMgTensorDescriptor_s* cutensorMgTensorDescriptor_t;

struct cutensorMgCopyDescriptor_s;
typedef struct cutensorMgCopyDescriptor_s* cutensorMgCopyDescriptor_t;

struct cutensorMgCopyPlan_s;
typedef struct cutensorMgCopyPlan_s* cutensorMgCopyPlan_t;

struct cutensorMgContractionDescriptor_s;
typedef struct cutensorMgContractionDescriptor_s* cutensorMgContractionDescriptor_t;

struct cutensorMgContractionFind_s;
typedef struct cutensorMgContractionFind_s* cutensorMgContractionFind_t;

struct cutensorMgContractionPlan_s;
typedef struct cutensorMgContractionPlan_s* cutensorMgContractionPlan_t;

typedef enum
{
    CUTENSORMG_ALGO_DEFAULT           = -1, ///< Lets the internal heuristic choose
} cutensorMgAlgo_t;

cutensorStatus_t
cutensorMgCreate(
    cutensorMgHandle_t* handle,
    uint32_t numDevices,
    const int32_t devices[]
);

cutensorStatus_t
cutensorMgDestroy(
    cutensorMgHandle_t handle
);

cutensorStatus_t
cutensorMgCreateTensorDescriptor(
    cutensorMgHandle_t handle,
    cutensorMgTensorDescriptor_t* desc,
    uint32_t numModes,
    const int64_t extent[],
    const int64_t elementStride[], // NULL -> dense
    const int64_t blockSize[], // NULL -> extent
    const int64_t blockStride[], // NULL -> elementStride
    const int32_t deviceCount[], // NULL -> 1
    uint32_t numDevices, const int32_t devices[],
    cudaDataType_t type
);

cutensorStatus_t
cutensorMgDestroyTensorDescriptor(
    cutensorMgTensorDescriptor_t desc
);

cutensorStatus_t
cutensorMgCreateCopyDescriptor(
    const cutensorMgHandle_t handle,
    cutensorMgCopyDescriptor_t *desc,
    const cutensorMgTensorDescriptor_t descDst, const int32_t modesDst[],
    const cutensorMgTensorDescriptor_t descSrc, const int32_t modesSrc[]);

cutensorStatus_t
cutensorMgDestroyCopyDescriptor(cutensorMgCopyDescriptor_t desc);

cutensorStatus_t
cutensorMgCopyGetWorkspace(
    const cutensorMgHandle_t handle,
    const cutensorMgCopyDescriptor_t desc,
    int64_t deviceWorkspaceSize[],
    int64_t* hostWorkspaceSize
);

cutensorStatus_t
cutensorMgCreateCopyPlan(
    const cutensorMgHandle_t handle,
    cutensorMgCopyPlan_t* plan,
    const cutensorMgCopyDescriptor_t desc,
    const int64_t deviceWorkspaceSize[],
    int64_t hostWorkspaceSize
);

cutensorStatus_t
cutensorMgDestroyCopyPlan(
    cutensorMgCopyPlan_t plan
);

cutensorStatus_t
cutensorMgCopy(
    const cutensorMgHandle_t handle,
    const cutensorMgCopyPlan_t plan,
    void* ptrDst[], // order from dst descriptor
    const void* ptrSrc[], // order from source descriptor
    void* deviceWorkspace[], // order from handle
    void* hostWorkspace,
    cudaStream_t streams[] // order from handle
);

cutensorStatus_t cutensorMgCreateContractionFind(const cutensorMgHandle_t handle,
                                             cutensorMgContractionFind_t* find,
                                             const cutensorMgAlgo_t algo);

cutensorStatus_t cutensorMgDestroyContractionFind(cutensorMgContractionFind_t find);

cutensorStatus_t
cutensorMgCreateContractionDescriptor(
    const cutensorMgHandle_t handle,
    cutensorMgContractionDescriptor_t* desc,
    const cutensorMgTensorDescriptor_t descA, const int32_t modesA[],
    const cutensorMgTensorDescriptor_t descB, const int32_t modesB[],
    const cutensorMgTensorDescriptor_t descC, const int32_t modesC[],
    const cutensorMgTensorDescriptor_t descD, const int32_t modesD[], // descD may be null
    cutensorComputeType_t compute);

cutensorStatus_t
cutensorMgDestroyContractionDescriptor(cutensorMgContractionDescriptor_t desc);

cutensorStatus_t
cutensorMgContractionGetWorkspace(
    const cutensorMgHandle_t handle,
    const cutensorMgContractionDescriptor_t desc,
    const cutensorMgContractionFind_t find,
    cutensorWorksizePreference_t preference,
    int64_t deviceWorkspaceSize[],
    int64_t* hostWorkspaceSize
);

cutensorStatus_t
cutensorMgCreateContractionPlan(
    const cutensorMgHandle_t handle,
    cutensorMgContractionPlan_t* plan,
    const cutensorMgContractionDescriptor_t desc,
    const cutensorMgContractionFind_t find,
    const int64_t deviceWorkspaceSize[],
    int64_t hostWorkspaceSize
);

cutensorStatus_t
cutensorMgDestroyContractionPlan(
    cutensorMgContractionPlan_t plan
);

/**
 * \param[out] hostWorkspace pointer to pinned host memory of size hostWorkspaceSize (see cutensorMgContractionGetWorkspace())
 */
cutensorStatus_t
cutensorMgContraction(
    const cutensorMgHandle_t handle,
    const cutensorMgContractionPlan_t plan,
    const void* alpha,
    const void* ptrA[],
    const void* ptrB[],
    const void* beta,
    const void* ptrC[],
    void* ptrD[],
    void* deviceWorkspace[], void* hostWorkspace,
    cudaStream_t streams[]
);

#if defined(__cplusplus)
}
#endif /* __cplusplus */
