/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2010-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#ifndef fatbinaryctl_INCLUDED
#define fatbinaryctl_INCLUDED

#ifndef __CUDA_INTERNAL_COMPILATION__
#include <stddef.h> /* for size_t */
#endif
#include "fatbinary.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * These are routines for controlling the fat binary.
 * An opaque handle is used.
 */
typedef struct fatBinaryCtlHandle *fatBinaryCtl_t;
typedef const struct fatBinaryCtlHandle *fatBinaryCtl_ct;
// !!! until driver sources are changed, do not require opaque type

typedef enum {
  FBCTL_ERROR_NONE = 0,
  FBCTL_ERROR_NULL,                      /* null pointer */
  FBCTL_ERROR_UNRECOGNIZED,              /* unrecognized kind */
  FBCTL_ERROR_NO_CANDIDATE,              /* no candidate found */
  FBCTL_ERROR_COMPILE_FAILED,            /* no candidate found */
  FBCTL_ERROR_INTERNAL,                  /* unexpected internal error */
  FBCTL_ERROR_COMPILER_LOAD_FAILED,      /* loading compiler library failed */
  FBCTL_ERROR_UNSUPPORTED_PTX_VERSION,   /* ptx version not supported by the compiler */
} fatBinaryCtlError_t;
extern const char* fatBinaryCtl_Errmsg (fatBinaryCtlError_t e);

/* Cannot change directly to opaque handle without causing warnings,
 * so add new CreateHandle routine and eventually switch everyone to it. */
extern fatBinaryCtlError_t fatBinaryCtl_Create (fatBinaryCtl_t *handle);
extern fatBinaryCtlError_t fatBinaryCtl_CreateHandle (fatBinaryCtl_t *handle);

extern void fatBinaryCtl_Delete (fatBinaryCtl_t handle);

/* Set (fatbin or elf) binary that we will search */
extern fatBinaryCtlError_t fatBinaryCtl_SetBinary (fatBinaryCtl_t handle, 
                                                   const void* binary);

/* Set target SM that we are looking for */
extern fatBinaryCtlError_t fatBinaryCtl_SetTargetSM (fatBinaryCtl_t handle,
                                                     unsigned int arch);

typedef enum {
  fatBinary_PreferBestCode,  /* default */
  fatBinary_AvoidPTX,        /* use sass if possible for compile-time savings */
  fatBinary_ForcePTX,        /* use ptx (mainly for testing) */
  fatBinary_JITIfNotMatch,   /* use ptx if arch doesn't match */
  fatBinary_PreferNvvm,      /* choose NVVM IR when available */
  fatBinary_LinkCompatible,  /* use sass if link-compatible */
  fatBinary_PreferMercury,   /* choose mercury over sass */
} fatBinary_CompilationPolicy;
/* Set policy for how we handle JIT compiles */
extern fatBinaryCtlError_t fatBinaryCtl_SetPolicy(fatBinaryCtl_t handle,
                                            fatBinary_CompilationPolicy policy);

/* Set ptxas options for JIT compiles */
extern fatBinaryCtlError_t fatBinaryCtl_SetPtxasOptions(fatBinaryCtl_t handle,
                                                        const char *options);

/* Set flags for fatbinary */
extern fatBinaryCtlError_t fatBinaryCtl_SetFlags (fatBinaryCtl_t handle,
                                                  long long flags);

/* Return identifier string for fatbinary */
extern fatBinaryCtlError_t fatBinaryCtl_GetIdentifier(fatBinaryCtl_ct handle,
                                                      const char **id);

/* Return ptxas options for fatbinary */
extern fatBinaryCtlError_t fatBinaryCtl_GetPtxasOptions(fatBinaryCtl_ct handle,
                                                        const char **options);

/* Return cicc options for fatbinary */
extern fatBinaryCtlError_t fatBinaryCtl_GetCiccOptions(fatBinaryCtl_ct handle,
                                                       const char **options);

/* Return whether fatbin has debug code (1 == true, 0 == false) */
extern fatBinaryCtlError_t fatBinaryCtl_HasDebug(fatBinaryCtl_ct handle,
                                                 int *debug);

/* Using the input values, pick the best candidate */
extern fatBinaryCtlError_t fatBinaryCtl_PickCandidate (fatBinaryCtl_t handle);

/* 
 * Using the previously chosen candidate, compile the code to elf,
 * returning elf image and size.
 * Note that because elf is allocated inside fatBinaryCtl, 
 * it will be freed when _Delete routine is called.
 */
extern fatBinaryCtlError_t fatBinaryCtl_Compile (fatBinaryCtl_t handle,
                                                 void* *elf, size_t *esize);

/*
 * Similar to fatBinaryCtl_Compile with extra support for
 * specifying the directory from where JIT compiler is to be loaded.
 */
extern fatBinaryCtlError_t fatBinaryCtl_Compile_WithJITDir (fatBinaryCtl_t handle,
                                                            void* *elf, size_t *esize,
                                                            const char* jitDir);

/* If *_Compile returned an error, get the error log */
extern fatBinaryCtlError_t fatBinaryCtl_GetCompileLog (fatBinaryCtl_t handle,
                                                       char **log);

/* Return the candidate found */
extern fatBinaryCtlError_t fatBinaryCtl_GetCandidate(fatBinaryCtl_ct handle, 
                                                     void **binary,
                                                     fatBinaryCodeKind *kind,
                                                     size_t *size);

#ifdef __cplusplus
}
#endif

#endif /* fatbinaryctl_INCLUDED */
