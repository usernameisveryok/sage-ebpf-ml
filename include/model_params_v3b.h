#pragma once

/*
 * model_params_v3b.h — Placeholder for V3b variant
 * Architecture: [12, 64, 64, 4]
 * Layer 0: MapLUT (same as V3)
 * Layer 1 & 2: Standard Int32 multiplication (NOT ternary)
 * Scale: 2^16 = 65536
 *
 * NOTE: Weight values are DUMMY ZEROS. Replace with actual trained
 *       values from the training script before deployment.
 */

#include <linux/types.h>

#define NUM_FEATURES        12
#define HIDDEN_SIZE         64
#define NUM_CLASSES          4
#define SCALE_FACTOR_BITS   16
#define SCALE_FACTOR        65536
#define NUM_BINS            512

#define CLASS_BENIGN       0
#define CLASS_DDOS         1
#define CLASS_PORTSCAN     2
#define CLASS_BRUTEFORCE   3

/* Early-exit confidence threshold (placeholder) */
#define EARLY_EXIT_THRESHOLD  20000

/* ------------------------------------------------------------------ */
/*  Per-feature quantization for MapLUT binning                        */
/* ------------------------------------------------------------------ */
static const __s32 feat_offset[12] = {
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
};

static const __u32 feat_shift[12] = {
    5, 15, 4, 19, 10, 7, 8, 8, 0, 11, 4, 0
};

/* ------------------------------------------------------------------ */
/*  Layer 0 bias [64] — added before LUT accumulation                  */
/* ------------------------------------------------------------------ */
static const __s32 bias_layer_0[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* ------------------------------------------------------------------ */
/*  Exit head bias [4] — for early-exit logit computation via MapLUT   */
/* ------------------------------------------------------------------ */
static const __s32 exit_bias[4] = { 0, 0, 0, 0 };

/* ------------------------------------------------------------------ */
/*  Layer 1: Standard Int32 weights [64 -> 64]                         */
/*  w1[i][j] = quantized weight for output neuron i, input j           */
/* ------------------------------------------------------------------ */
static const __s32 w1[64][64] = { {0} };

static const __s32 bias_layer_1[64] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

/* ------------------------------------------------------------------ */
/*  Layer 2: Standard Int32 weights [4 -> 64]                          */
/*  w2[i][j] = quantized weight for output class i, input j            */
/* ------------------------------------------------------------------ */
static const __s32 w2[4][64] = { {0} };

static const __s32 bias_layer_2[4] = { 0, 0, 0, 0 };
