// Test-only, MPL-2.0. SHA-256 subset of BCrypt for the POSIX shim; see windows.h.
#pragma once
#include "windows.h"
typedef void* BCRYPT_ALG_HANDLE;
typedef void* BCRYPT_HASH_HANDLE;
#define BCRYPT_SHA256_ALGORITHM L"SHA256"
#define BCRYPT_OBJECT_LENGTH L"ObjectLength"
NTSTATUS BCryptOpenAlgorithmProvider(BCRYPT_ALG_HANDLE*, const wchar_t* algorithm, const wchar_t* implementation, ULONG flags);
NTSTATUS BCryptGetProperty(void* object, const wchar_t* property, PUCHAR output, ULONG outputBytes, ULONG* result, ULONG flags);
NTSTATUS BCryptCloseAlgorithmProvider(BCRYPT_ALG_HANDLE, ULONG flags);
NTSTATUS BCryptCreateHash(BCRYPT_ALG_HANDLE, BCRYPT_HASH_HANDLE*, PUCHAR object, ULONG objectBytes, PUCHAR secret, ULONG secretBytes, ULONG flags);
NTSTATUS BCryptDestroyHash(BCRYPT_HASH_HANDLE);
NTSTATUS BCryptHashData(BCRYPT_HASH_HANDLE, PUCHAR input, ULONG bytes, ULONG flags);
NTSTATUS BCryptFinishHash(BCRYPT_HASH_HANDLE, PUCHAR output, ULONG bytes, ULONG flags);
