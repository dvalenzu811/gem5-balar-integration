// Copyright 2009-2026 NTESS. Under the terms
// of Contract DE-NA0003525 with NTESS, the U.S.
// Government retains certain rights in this software.
//
// Copyright (c) 2009-2026, NTESS
// All rights reserved.
//
// Portions are copyright of other developers:
// See the file CONTRIBUTORS.TXT in the top level directory
// of the distribution for more information.
//
// This file is part of the SST software package. For license
// information, see the LICENSE file in the top level directory of the
// distribution.

#ifndef __BALAR_VANADIS_H__
#define __BALAR_VANADIS_H__

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#define LOG_LEVEL_INFO 20
#define LOG_LEVEL_DEBUG 15
#define LOG_LEVEL_WARNING 10
#define LOG_LEVEL_ERROR 0

// Depends on arch, mips use uint32_t, riscv get uint64_t
typedef uint64_t Addr_t;

/*
 * BSC gem5 Full-System map.
 *
 * Gem5Balar owns the physical MMIO range:
 *     0x10020000 - 0x10020fff
 *
 * CUDA packets live in normal Linux RAM. Their physical
 * addresses are obtained through /proc/self/pagemap before
 * being sent to Gem5Balar.
 */
#define BALAR_MMIO_PHYS        0x10020000ULL
#define BALAR_SHARED_SIZE      0x01000000ULL
#define BALAR_STAGING_OFFSET   0x00001000ULL
#define BALAR_STAGING_SIZE     (BALAR_SHARED_SIZE - BALAR_STAGING_OFFSET)

static volatile Addr_t* g_balarBaseAddr = NULL;

static uint8_t* g_shared_base = NULL;
static uint8_t* g_scratch_mem = NULL;
static uint8_t* g_staging_mem = NULL;

static uint64_t g_scratch_phys = 0;
static size_t g_page_size = 4096;

static int g_devmem_fd = -1;
static int g_pagemap_fd = -1;

static int32_t g_debug_level = LOG_LEVEL_INFO;

#endif
