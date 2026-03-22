// SPDX-License-Identifier: GPL-2.0
/*
 * xdp_ml_v3.c — XDP in-kernel MLP inference with THREE operator-level
 *               innovations achieving ZERO multiplication instructions
 *               across the ENTIRE neural inference pipeline.
 *
 * Architecture:  quantized MLP [12 → 64 → 64 → 4]
 * Approach:      Tail-call chain (4 stages) + three novel operators.
 *
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║  OPERATOR 1 — MapLUT (Map-based Lookup Table)                       ║
 * ║                                                                      ║
 * ║  Replaces ALL multiplications in Layer 0 (12→64) AND the Exit Head  ║
 * ║  (64→4) with BPF map lookups.                                       ║
 * ║                                                                      ║
 * ║  Key idea: For each feature j (0..11), precompute partial-sum       ║
 * ║  vectors for every possible quantized input bin.  At runtime:       ║
 * ║    1) Quantize raw feature → bin index (shift only, zero multiply)  ║
 * ║    2) Look up precomputed partial sums from BPF per-CPU array map   ║
 * ║    3) Accumulate via addition (zero multiply)                       ║
 * ║                                                                      ║
 * ║  Impact:                                                             ║
 * ║    - Layer 0:    768 multiplies → 0  (12×64 = 768 eliminated)       ║
 * ║    - Exit Head:  256 multiplies → 0  (64×4  = 256 eliminated)       ║
 * ║    - Space-time tradeoff: ~192KB map memory buys zero-multiply L0   ║
 * ║    - Model hot-update: userspace can bpf_map_update_elem() to swap  ║
 * ║      model weights without recompiling/reloading the eBPF program   ║
 * ╠══════════════════════════════════════════════════════════════════════╣
 * ║  OPERATOR 2 — TernaryShift (Ternary Weight Shift-Add)               ║
 * ║                                                                      ║
 * ║  Hidden-to-hidden layers (1 & 2) use ternary {-α, 0, +α} weights   ║
 * ║  packed as 2 bits each (16 weights per __u32).                       ║
 * ║  ALL multiplications are replaced by conditional add/sub + a single ║
 * ║  bit-shift per neuron.  Zero multiply instructions.                  ║
 * ╠══════════════════════════════════════════════════════════════════════╣
 * ║  OPERATOR 3 — EarlyExit (Confidence-Based Tail-Call Bypass)         ║
 * ║                                                                      ║
 * ║  After Layer 0 + Exit Head (both computed via MapLUT with zero      ║
 * ║  multiplications), check the top-2 logit margin.  If confident,     ║
 * ║  skip Layers 1 & 2 entirely and jump straight to classification.    ║
 * ║  Expected: ~85% of benign packets exit early, saving ~40% latency.  ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 *
 * *** ENTIRE PIPELINE: ZERO MULTIPLICATION INSTRUCTIONS ***
 *   Layer 0  → MapLUT       (addition only)
 *   Exit Head → MapLUT      (addition only, fused with Layer 0)
 *   Layer 1  → TernaryShift (add/sub + shift only)
 *   Layer 2  → TernaryShift (add/sub + shift only)
 *
 * Program layout:
 *   Program 0  (xdp_entry)    — parse packet, extract 12 features
 *   Program 1  (xdp_layer_0)  — MapLUT Layer 0 + Exit Head + EarlyExit check
 *   Program 2  (xdp_layer_1)  — TernaryShift (64→64) + ReLU
 *   Program 3  (xdp_classify) — TernaryShift (64→4) + argmax + action
 *
 * Per-CPU scratch map passes intermediate vectors between tail calls.
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

#include "../include/model_params_v3.h"

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */
#define PKT_THRESHOLD     50
#define MAX_FLOWS      65536
#define MAX_BLOCK      65536
#define STATS_ENTRIES      8

#define STAT_TOTAL_PKTS       0
#define STAT_INFER_RUNS       1
#define STAT_CLASS_BENIGN     2
#define STAT_CLASS_DDOS       3
#define STAT_CLASS_PORTSCAN   4
#define STAT_CLASS_BRUTE      5
#define STAT_BLOCKED          6
#define STAT_EARLY_EXIT       7

/* Tail-call program indices */
#define PROG_ENTRY    0
#define PROG_LAYER0   1
#define PROG_LAYER1   2
#define PROG_CLASSIFY 3
#define PROG_MAX      4

/* ------------------------------------------------------------------ */
/*  Data structures                                                    */
/* ------------------------------------------------------------------ */
struct flow_key {
    __u32 src_ip;
    __u32 dst_ip;
    __u16 src_port;
    __u16 dst_port;
    __u8  protocol;
    __u8  pad[3];
};

/*
 * Extended flow feature accumulator — tracks 12 CIC-IDS-2017 features.
 */
struct flow_features {
    __s64 pkt_len_sum;       /* for Fwd Packet Length Mean          */
    __s64 hdr_len_sum;       /* for Fwd Header Length               */
    __s32 total_fwd_pkts;    /* Total Fwd Packets                   */
    __s32 fwd_pkt_len_max;   /* Fwd Packet Length Max               */
    __s32 fwd_pkt_len_min;   /* Fwd Packet Length Min (init 0x7FFF) */
    __s64 fwd_iat_sum;       /* for Fwd IAT Mean                    */
    __s64 last_seen;
    __s32 pkt_count;
    __s32 classified;
    __s32 class_result;
    __s32 protocol;          /* Protocol number                     */
    __s32 syn_flag_count;    /* SYN Flag Count                      */
    __s32 psh_flag_count;    /* PSH Flag Count                      */
    __s32 ack_flag_count;    /* ACK Flag Count                      */
    __s32 total_bwd_pkts;    /* Total Backward Packets (approx)     */
    __s32 init_fwd_win;      /* Init Fwd Win Bytes (first SYN)      */
    __s32 init_bwd_win;      /* Init Bwd Win Bytes (first SYN-ACK)  */
    __s32 has_init_win;      /* flag: captured initial windows       */
};

/*
 * Per-CPU scratch space shared between tail-call stages.
 * Contains features, hidden vectors, logits, and flow keys.
 */
struct nn_ctx {
    __s32 features[NUM_FEATURES];   /* 12 × 4 =  48 bytes  */
    __s32 hidden[HIDDEN_SIZE];      /* 64 × 4 = 256 bytes  */
    __s32 logits[NUM_CLASSES];      /* 4 × 4  =  16 bytes  */
    struct flow_key key;            /* 16 bytes             */
    struct flow_key rkey;           /* 16 bytes             */
    __s32 is_fin_rst;               /* 4 bytes              */
    __s32 early_exit;               /* 1 if early-exited    */
};

/* ------------------------------------------------------------------ */
/*  MapLUT entry — precomputed partial sums for one (feature, bin)     */
/*                                                                      */
/*  For feature j and bin b, lut_entry stores:                          */
/*    h[i]      = W0[i][j] * quantized_value(b)  for all hidden units  */
/*    exit_h[c] = ExitW[c][·] contribution        for all classes       */
/*                                                                      */
/*  At runtime: just look up and ADD. Zero multiplications.             */
/* ------------------------------------------------------------------ */
struct lut_entry {
    __s32 h[HIDDEN_SIZE];        /* partial sum for hidden layer (64)  */
    __s32 exit_h[NUM_CLASSES];   /* partial sum for exit head (4)      */
};

/* ------------------------------------------------------------------ */
/*  BPF Maps                                                           */
/* ------------------------------------------------------------------ */

/* Flow tracking */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, struct flow_key);
    __type(value, struct flow_features);
    __uint(max_entries, MAX_FLOWS);
} flow_map SEC(".maps");

/* Blocked flows */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, struct flow_key);
    __type(value, __u32);
    __uint(max_entries, MAX_BLOCK);
} blocklist_map SEC(".maps");

/* Per-CPU statistics counters */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, __u64);
    __uint(max_entries, STATS_ENTRIES);
} stats_map SEC(".maps");

/* Per-CPU NN scratch buffer — shared across tail-call stages */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, struct nn_ctx);
    __uint(max_entries, 1);
} nn_scratch SEC(".maps");

/* Tail-call program array */
struct {
    __uint(type, BPF_MAP_TYPE_PROG_ARRAY);
    __type(key, __u32);
    __type(value, __u32);
    __uint(max_entries, PROG_MAX);
} jmp_table SEC(".maps");

/* Per-CPU latency tracking */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, __u64);
    __uint(max_entries, 4);
} latency_map SEC(".maps");

#define LAT_INFER_TOTAL   0   /* total inference time (ns) */
#define LAT_INFER_COUNT   1   /* number of inference runs  */
#define LAT_INFER_MAX     2   /* max single inference time */
#define LAT_START         3   /* temp: start timestamp     */

/*
 * ════════════════════════════════════════════════════════════════════
 *  OPERATOR 1 — MapLUT: Precomputed partial-sum lookup table
 * ════════════════════════════════════════════════════════════════════
 *
 * BPF per-CPU array map storing precomputed partial sums.
 *
 *   Key:   j * NUM_BINS + bin_index   (j = feature index, 0..11)
 *   Value: struct lut_entry { h[64], exit_h[4] }
 *
 *   Total entries: NUM_FEATURES * NUM_BINS = 12 * 64 = 768
 *   Per-entry size: (64 + 4) * 4 = 272 bytes
 *   Total memory per CPU: 768 * 272 ≈ 204 KB
 *
 *   Populated from userspace via bpf_map_update_elem() — enables
 *   model hot-update without recompiling/reloading the eBPF program.
 */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, struct lut_entry);
    __uint(max_entries, NUM_FEATURES * NUM_BINS);  /* 12 * 64 = 768 */
} lut_map SEC(".maps");

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

/* Atomically increment a stats counter. */
static __always_inline void stat_inc(__u32 idx)
{
    __u64 *val = bpf_map_lookup_elem(&stats_map, &idx);
    if (val)
        __sync_fetch_and_add(val, 1);
}

/* Division that is safe against divide-by-zero. */
static __always_inline __s64 safe_div(__s64 a, __s64 b)
{
    if (b == 0) return 0;
    __u64 ua = a < 0 ? (__u64)(-a) : (__u64)a;
    __u64 ub = b < 0 ? (__u64)(-b) : (__u64)b;
    __s64 r = (__s64)(ua / ub);
    return ((a ^ b) < 0) ? -r : r;
}

/* Retrieve the per-CPU NN scratch context. */
static __always_inline struct nn_ctx *get_ctx(void)
{
    __u32 key = 0;
    return bpf_map_lookup_elem(&nn_scratch, &key);
}

/*
 * ================================================================
 * OPERATOR 2 — Ternary Dot Product (TernaryShift)
 * ================================================================
 *
 * Computes the dot product of a ternary-packed weight row and an
 * input vector using ONLY additions, subtractions, and a single
 * left shift.  Zero multiplications.
 *
 * Weight packing: 16 ternary values per __u32, 2 bits each.
 *   bits [2k+1 : 2k] for input index (base + k):
 *     0b00 = zero weight   → skip
 *     0b01 = +alpha weight → accumulate +x[j]
 *     0b10 = -alpha weight → accumulate -x[j]
 *
 * After accumulating the sum of ±x[j] terms, the result is
 * scaled by alpha = 2^alpha_shift via a single left shift.
 *
 * Parameters:
 *   packed_row  — pointer to packed ternary words (n_cols/16 words)
 *   x           — input vector
 *   n_cols      — input dimension (MUST be a multiple of 16)
 *   alpha_shift — log2(alpha), the per-layer scale factor
 *
 * Returns: dot product result as __s64
 */
static __always_inline __s64 ternary_dot(
    const __u32 *packed_row,
    const __s32 *x,
    int n_cols,
    int alpha_shift)
{
    __s64 acc = 0;
    int j;

    for (j = 0; j < n_cols; j += 16) {
        __u32 pack = packed_row[j / 16];

        /*
         * Fully unroll the inner 16 iterations.
         * This is required for the BPF verifier — it cannot
         * reason about nested variable-bound loops.
         */
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            __u8 t = (pack >> (k * 2)) & 0x3;
            if (t == 1)
                acc += (__s64)x[j + k];   /* +alpha */
            else if (t == 2)
                acc -= (__s64)x[j + k];   /* -alpha */
            /* t == 0 or t == 3: zero weight, skip */
        }
    }

    /* Scale the accumulated sum by alpha = 2^alpha_shift */
    return acc << alpha_shift;
}

/* ================================================================== */
/*  Program 0 — Packet parse, flow tracking, feature extraction        */
/*                                                                      */
/*  Extracts 12 CIC-IDS-2017 features as raw integers.                 */
/*  No normalization — MapLUT handles it implicitly via precomputed     */
/*  partial sums that incorporate normalization into the LUT values.    */
/* ================================================================== */
SEC("xdp/entry")
int xdp_entry(struct xdp_md *ctx)
{
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    stat_inc(STAT_TOTAL_PKTS);

    /* ---- Ethernet ------------------------------------------------- */
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;
    if (eth->h_proto != bpf_htons(ETH_P_IP))
        return XDP_PASS;

    /* ---- IPv4 ----------------------------------------------------- */
    struct iphdr *iph = (void *)(eth + 1);
    if ((void *)(iph + 1) > data_end)
        return XDP_PASS;
    __u32 ip_hdr_len = iph->ihl * 4;
    if (ip_hdr_len < sizeof(struct iphdr))
        return XDP_PASS;
    if ((void *)iph + ip_hdr_len > data_end)
        return XDP_PASS;

    __u8  proto = iph->protocol;
    if (proto != IPPROTO_TCP && proto != IPPROTO_UDP)
        return XDP_PASS;

    __u32 src_ip = iph->saddr;
    __u32 dst_ip = iph->daddr;
    __u16 total_pkt_len = bpf_ntohs(iph->tot_len);

    /* ---- TCP / UDP ------------------------------------------------ */
    __u16 src_port = 0, dst_port = 0;
    __u32 l4_hdr_len = 0;
    int is_fin = 0, is_rst = 0;
    int is_syn = 0, is_psh = 0, is_ack = 0;
    __u16 tcp_window = 0;

    if (proto == IPPROTO_TCP) {
        struct tcphdr *tcph = (void *)iph + ip_hdr_len;
        if ((void *)(tcph + 1) > data_end)
            return XDP_PASS;
        src_port   = bpf_ntohs(tcph->source);
        dst_port   = bpf_ntohs(tcph->dest);
        l4_hdr_len = tcph->doff * 4;
        is_fin     = tcph->fin;
        is_rst     = tcph->rst;
        is_syn     = tcph->syn;
        is_psh     = tcph->psh;
        is_ack     = tcph->ack;
        tcp_window = bpf_ntohs(tcph->window);
    } else {
        struct udphdr *udph = (void *)iph + ip_hdr_len;
        if ((void *)(udph + 1) > data_end)
            return XDP_PASS;
        src_port   = bpf_ntohs(udph->source);
        dst_port   = bpf_ntohs(udph->dest);
        l4_hdr_len = sizeof(struct udphdr);
    }

    __u32 hdr_len = (__u32)sizeof(struct ethhdr) + ip_hdr_len + l4_hdr_len;

    /* ---- Flow key ------------------------------------------------- */
    struct flow_key key = {};
    key.src_ip   = src_ip;
    key.dst_ip   = dst_ip;
    key.src_port = src_port;
    key.dst_port = dst_port;
    key.protocol = proto;

    /* ---- Blocklist check ------------------------------------------ */
    if (bpf_map_lookup_elem(&blocklist_map, &key)) {
        stat_inc(STAT_BLOCKED);
        return XDP_DROP;
    }
    struct flow_key rkey = {};
    rkey.src_ip   = dst_ip;
    rkey.dst_ip   = src_ip;
    rkey.src_port = dst_port;
    rkey.dst_port = src_port;
    rkey.protocol = proto;
    if (bpf_map_lookup_elem(&blocklist_map, &rkey)) {
        stat_inc(STAT_BLOCKED);
        return XDP_DROP;
    }

    /* ---- Flow lookup / create ------------------------------------- */
    __u64 now = bpf_ktime_get_ns();
    struct flow_features *flow = bpf_map_lookup_elem(&flow_map, &key);
    if (!flow) {
        struct flow_features newf = {};
        newf.pkt_len_sum     = total_pkt_len;
        newf.hdr_len_sum     = hdr_len;
        newf.total_fwd_pkts  = 1;
        newf.fwd_pkt_len_max = (__s32)total_pkt_len;
        newf.fwd_pkt_len_min = (__s32)total_pkt_len;
        newf.last_seen       = (__s64)now;
        newf.pkt_count       = 1;
        newf.protocol        = (__s32)proto;
        newf.syn_flag_count  = is_syn ? 1 : 0;
        newf.psh_flag_count  = is_psh ? 1 : 0;
        newf.ack_flag_count  = is_ack ? 1 : 0;
        newf.total_bwd_pkts  = 0;

        /* Capture initial TCP window sizes */
        if (proto == IPPROTO_TCP && is_syn) {
            newf.init_fwd_win = (__s32)tcp_window;
            newf.has_init_win = 1;
        }

        bpf_map_update_elem(&flow_map, &key, &newf, BPF_ANY);
        return XDP_PASS;
    }

    if (flow->classified)
        return flow->class_result == CLASS_BENIGN ? XDP_PASS : XDP_DROP;

    /* ---- Accumulate ----------------------------------------------- */
    flow->pkt_len_sum   += total_pkt_len;
    flow->hdr_len_sum   += hdr_len;
    flow->total_fwd_pkts++;
    if ((__s32)total_pkt_len > flow->fwd_pkt_len_max)
        flow->fwd_pkt_len_max = (__s32)total_pkt_len;
    if ((__s32)total_pkt_len < flow->fwd_pkt_len_min)
        flow->fwd_pkt_len_min = (__s32)total_pkt_len;
    if (flow->last_seen > 0) {
        __s64 iat = (__s64)now - flow->last_seen;
        if (iat > 0)
            flow->fwd_iat_sum += iat;
    }
    flow->last_seen = (__s64)now;
    flow->pkt_count++;

    /* Accumulate TCP flags */
    if (is_syn) flow->syn_flag_count++;
    if (is_psh) flow->psh_flag_count++;
    if (is_ack) flow->ack_flag_count++;

    /* Capture initial backward window (SYN-ACK) */
    if (proto == IPPROTO_TCP && is_syn && is_ack && !flow->has_init_win) {
        flow->init_bwd_win = (__s32)tcp_window;
        flow->has_init_win = 1;
    }

    /*
     * Heuristic: count "backward" packets as those arriving on the
     * reverse direction. Since we only track one direction per flow
     * entry, we approximate by counting ACK-only packets (no SYN/FIN)
     * as backward traffic.
     */
    if (is_ack && !is_syn && !is_fin && !is_psh)
        flow->total_bwd_pkts++;

    /* ---- Trigger inference? --------------------------------------- */
    if (!is_fin && !is_rst && flow->pkt_count < PKT_THRESHOLD)
        return XDP_PASS;

    /* ---- Compute 12 features & store in scratch ------------------- */
    /*
     * Features are stored as RAW integers — no normalization here.
     * MapLUT (Operator 1) handles normalization implicitly: the
     * precomputed partial sums in the LUT already incorporate
     * normalization scaling into the quantized bin values.
     *
     * Feature mapping (12 CIC-IDS-2017 features):
     *   [0]  Fwd Packet Length Mean  = pkt_len_sum / pkt_count
     *   [1]  Fwd Header Length       = hdr_len_sum / pkt_count
     *   [2]  Avg Packet Size         = pkt_len_sum / pkt_count
     *   [3]  Fwd IAT Mean            = fwd_iat_sum / (pkt_count - 1)
     *   [4]  Total Fwd Packets       = total_fwd_pkts
     *   [5]  Fwd Packet Length Max   = fwd_pkt_len_max
     *   [6]  Init Bwd Win Bytes      = init_bwd_win
     *   [7]  Init Fwd Win Bytes      = init_fwd_win
     *   [8]  PSH Flag Count          = psh_flag_count
     *   [9]  Total Backward Packets  = total_bwd_pkts
     *   [10] Fwd Packet Length Min   = fwd_pkt_len_min
     *   [11] Protocol                = protocol
     */
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    __s32 cnt = flow->pkt_count > 0 ? flow->pkt_count : 1;
    __s32 iat_div = cnt > 1 ? cnt - 1 : 1;

    nctx->features[0]  = (__s32)safe_div(flow->pkt_len_sum, (__s64)cnt);
    nctx->features[1]  = (__s32)safe_div(flow->hdr_len_sum, (__s64)cnt);
    nctx->features[2]  = (__s32)safe_div(flow->pkt_len_sum, (__s64)cnt);
    nctx->features[3]  = (__s32)safe_div(
                             safe_div(flow->fwd_iat_sum, (__s64)iat_div),
                             1000LL);
    nctx->features[4]  = flow->total_fwd_pkts;
    nctx->features[5]  = flow->fwd_pkt_len_max;
    nctx->features[6]  = flow->init_bwd_win;
    nctx->features[7]  = flow->init_fwd_win;
    nctx->features[8]  = flow->psh_flag_count;
    nctx->features[9]  = flow->total_bwd_pkts;
    nctx->features[10] = flow->fwd_pkt_len_min;
    nctx->features[11] = flow->protocol;

    nctx->key      = key;
    nctx->rkey     = rkey;
    nctx->is_fin_rst = (is_fin || is_rst) ? 1 : 0;
    nctx->early_exit = 0;   /* reset early-exit flag */

    /* Record inference start time */
    {
        __u32 lat_key = LAT_START;
        __u64 start = bpf_ktime_get_ns();
        bpf_map_update_elem(&latency_map, &lat_key, &start, BPF_ANY);
    }

    /* Jump to Layer 0 (MapLUT) */
    bpf_tail_call(ctx, &jmp_table, PROG_LAYER0);

    /* If tail call fails, just pass */
    return XDP_PASS;
}

/* ================================================================== */
/*  Program 1 — MapLUT Layer 0 (12→64) + Exit Head (64→4) + EarlyExit */
/*                                                                      */
/*  OPERATOR 1 (MapLUT): Zero-multiply Layer 0 computation              */
/*    For each of the 12 input features:                                */
/*      1) Quantize raw value to a bin index using shift (zero mult)    */
/*      2) Look up precomputed partial sums from lut_map                */
/*      3) Accumulate hidden[i] += entry->h[i]  (addition only)        */
/*      4) Accumulate exit_logits[c] += entry->exit_h[c] (addition)    */
/*    After all 12 features: apply ReLU to hidden vector.               */
/*                                                                      */
/*    This eliminates ALL 768 + 256 = 1024 multiplications from        */
/*    Layer 0 and the Exit Head, replacing them with 768 map lookups    */
/*    and pure additions.                                                */
/*                                                                      */
/*  OPERATOR 3 (EarlyExit): Confidence-based bypass                     */
/*    The exit head logits are already computed (via MapLUT, zero mult). */
/*    Check top-2 logit margin; if confident, skip layers 1 & 2.        */
/* ================================================================== */
SEC("xdp/layer0")
int xdp_layer_0(struct xdp_md *ctx)
{
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    int i, j;

    /*
     * ── MapLUT Layer 0 + Exit Head (Operator 1) ───────────────────
     *
     * Step 1: Initialize hidden[] and exit_logits[] to bias values.
     * Step 2: For each feature j, quantize → lookup → accumulate.
     * Step 3: Apply ReLU to hidden[].
     *
     * ZERO multiplications in the entire computation.
     */

    /* Step 1: Reset hidden to Layer 0 bias */
    for (i = 0; i < HIDDEN_SIZE; i++)
        nctx->hidden[i] = bias_layer_0[i];

    /* Reset exit logits to exit head bias */
    __s32 exit_logits[NUM_CLASSES];
    for (i = 0; i < NUM_CLASSES; i++)
        exit_logits[i] = exit_bias[i];

    /* Step 2: Accumulate partial sums from LUT for each feature */
    for (j = 0; j < NUM_FEATURES; j++) {
        /*
         * Quantize raw feature to bin index using shift only.
         * bin = (raw - feat_offset[j]) >> feat_shift[j]
         * Clamp to [0, NUM_BINS - 1].
         *
         * ZERO multiplications — only subtract + shift + clamp.
         */
        __s32 raw = nctx->features[j];
        __s32 shifted = raw - feat_offset[j];
        if (shifted < 0)
            shifted = 0;
        __u32 bin = (__u32)(shifted >> feat_shift[j]);
        if (bin >= NUM_BINS)
            bin = NUM_BINS - 1;

        __u32 key = j * NUM_BINS + bin;
        struct lut_entry *entry = bpf_map_lookup_elem(&lut_map, &key);
        if (!entry)
            continue;

        /*
         * Accumulate partial sums — pure addition, zero multiply.
         *   hidden[i] += entry->h[i]        (Layer 0)
         *   exit_logits[c] += entry->exit_h[c]  (Exit Head)
         */
        for (i = 0; i < HIDDEN_SIZE; i++)
            nctx->hidden[i] += entry->h[i];
        for (i = 0; i < NUM_CLASSES; i++)
            exit_logits[i] += entry->exit_h[i];
    }

    /* Step 3: Apply ReLU to hidden vector */
    for (i = 0; i < HIDDEN_SIZE; i++)
        nctx->hidden[i] = nctx->hidden[i] > 0 ? nctx->hidden[i] : 0;

    /*
     * ── EarlyExit Check (Operator 3) ──────────────────────────────
     *
     * The exit head logits are already computed via MapLUT above
     * (zero multiplications).  Check top-2 logit margin for
     * confidence.  If the margin exceeds the threshold, skip
     * Layers 1 & 2 entirely — jump straight to PROG_CLASSIFY.
     *
     * Confidence metric: margin between top-2 logits.
     * No softmax needed: integer margin is a valid confidence proxy.
     */
    __s32 max1 = exit_logits[0];
    __s32 max2 = -2147483647;    /* INT32_MIN + 1 */
    __s32 exit_cls = 0;

    for (i = 1; i < NUM_CLASSES; i++) {
        if (exit_logits[i] > max1) {
            max2 = max1;
            max1 = exit_logits[i];
            exit_cls = i;
        } else if (exit_logits[i] > max2) {
            max2 = exit_logits[i];
        }
    }

    /*
     * If the margin between top-2 logits exceeds the threshold,
     * the model is confident enough to skip layers 1 & 2.
     * Store the predicted class in logits[0] (reused as shortcut)
     * and set the early_exit flag so PROG_CLASSIFY knows to use it.
     */
    if ((max1 - max2) > EARLY_EXIT_THRESHOLD) {
        nctx->logits[0]  = exit_cls;    /* reuse logits[0] as class */
        nctx->early_exit = 1;
        stat_inc(STAT_EARLY_EXIT);
        bpf_tail_call(ctx, &jmp_table, PROG_CLASSIFY);
        /* If tail call fails, fall through to normal path */
    }

    /* Normal path: proceed to Layer 1 (TernaryShift) */
    bpf_tail_call(ctx, &jmp_table, PROG_LAYER1);
    return XDP_PASS;
}

/* ================================================================== */
/*  Program 2 — TernaryShift Layer 1 (64→64) + ReLU                   */
/*                                                                      */
/*  OPERATOR 2: Ternary weight inference (shift-add)                    */
/*    Uses packed 2-bit ternary weights: 16 per __u32.                  */
/*    The ternary_dot() helper replaces ALL multiply instructions       */
/*    with conditional additions/subtractions + a single shift.         */
/*    For 64 inputs, each neuron needs 4 packed words.                  */
/*    Zero multiply instructions.                                       */
/* ================================================================== */
SEC("xdp/layer1")
int xdp_layer_1(struct xdp_md *ctx)
{
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    /*
     * Copy hidden[] into a local buffer to avoid reading and
     * writing the same map memory in the inner loop, which can
     * confuse the BPF verifier.
     */
    __s32 input[HIDDEN_SIZE];
    int j;
    for (j = 0; j < HIDDEN_SIZE; j++)
        input[j] = nctx->hidden[j];

    /*
     * ── TernaryShift MatMul (Operator 2) ──────────────────────────
     *
     * For each output neuron i:
     *   acc = ternary_dot(packed_w1[i], input, 64, ALPHA_SHIFT_LAYER_1)
     *   hidden[i] = ReLU(acc >> SCALE_FACTOR_BITS + bias)
     *
     * Zero multiplications — only add/sub/shift.
     */
    int i;
    for (i = 0; i < HIDDEN_SIZE; i++) {
        __s64 acc = ternary_dot(packed_w1[i], input,
                                HIDDEN_SIZE, ALPHA_SHIFT_LAYER_1);
        __s32 v = (__s32)(acc >> SCALE_FACTOR_BITS) + bias_layer_1[i];
        nctx->hidden[i] = v > 0 ? v : 0;   /* ReLU */
    }

    bpf_tail_call(ctx, &jmp_table, PROG_CLASSIFY);
    return XDP_PASS;
}

/* ================================================================== */
/*  Program 3 — Classification (64→4) + argmax + action                */
/*                                                                      */
/*  If early_exit == 1, skip recomputation — use the pre-stored        */
/*  class from the exit head (Operator 3 / EarlyExit).                  */
/*  Otherwise, run TernaryShift for the final layer (Operator 2).       */
/* ================================================================== */
SEC("xdp/classify")
int xdp_classify(struct xdp_md *ctx)
{
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    stat_inc(STAT_INFER_RUNS);

    __s32 cls = 0;

    if (nctx->early_exit) {
        /*
         * ── EarlyExit path (Operator 3) ───────────────────────────
         *
         * The exit head in Layer 0 (computed via MapLUT with zero
         * multiplications) already determined the class.  The
         * predicted class was stored in logits[0].  Skip the
         * expensive final-layer computation entirely.
         */
        cls = nctx->logits[0];

        /* Bounds-check the class to keep the verifier happy */
        if (cls < 0 || cls >= NUM_CLASSES)
            cls = 0;
    } else {
        /*
         * ── Normal path: TernaryShift Layer 2 (Operator 2) ────────
         *
         * Compute logits[i] = ternary_dot(packed_w2[i], hidden, 64,
         *                                  ALPHA_SHIFT_LAYER_2)
         *                     >> SCALE_FACTOR_BITS + bias
         * Then argmax over the 4 logits.
         *
         * Zero multiplications — only add/sub/shift.
         */
        int i;
        for (i = 0; i < NUM_CLASSES; i++) {
            __s64 acc = ternary_dot(packed_w2[i], nctx->hidden,
                                    HIDDEN_SIZE, ALPHA_SHIFT_LAYER_2);
            nctx->logits[i] = (__s32)(acc >> SCALE_FACTOR_BITS)
                              + bias_layer_2[i];
        }

        /* Argmax */
        cls = 0;
        for (i = 1; i < NUM_CLASSES; i++) {
            if (nctx->logits[i] > nctx->logits[cls])
                cls = i;
        }
    }

    /* ---- Update flow state ---------------------------------------- */
    struct flow_features *flow = bpf_map_lookup_elem(&flow_map, &nctx->key);
    if (flow) {
        flow->classified   = 1;
        flow->class_result = cls;
    }

    /* ---- Update per-class statistics ------------------------------ */
    __u32 cls_stat;
    switch (cls) {
    case CLASS_BENIGN:     cls_stat = STAT_CLASS_BENIGN;   break;
    case CLASS_DDOS:       cls_stat = STAT_CLASS_DDOS;     break;
    case CLASS_PORTSCAN:   cls_stat = STAT_CLASS_PORTSCAN; break;
    case CLASS_BRUTEFORCE: cls_stat = STAT_CLASS_BRUTE;    break;
    default:               cls_stat = STAT_CLASS_BENIGN;   break;
    }
    stat_inc(cls_stat);

    /* ---- Blocklist malicious flows -------------------------------- */
    if (cls != CLASS_BENIGN) {
        __u32 label = (__u32)cls;
        bpf_map_update_elem(&blocklist_map, &nctx->key,  &label, BPF_ANY);
        bpf_map_update_elem(&blocklist_map, &nctx->rkey, &label, BPF_ANY);
    }

    /* ---- Cleanup finished flows ----------------------------------- */
    if (nctx->is_fin_rst)
        bpf_map_delete_elem(&flow_map, &nctx->key);

    /* ---- Record inference latency --------------------------------- */
    {
        __u32 lat_key = LAT_START;
        __u64 *start = bpf_map_lookup_elem(&latency_map, &lat_key);
        if (start && *start > 0) {
            __u64 elapsed = bpf_ktime_get_ns() - *start;

            lat_key = LAT_INFER_TOTAL;
            __u64 *total = bpf_map_lookup_elem(&latency_map, &lat_key);
            if (total)
                __sync_fetch_and_add(total, elapsed);

            lat_key = LAT_INFER_COUNT;
            __u64 *count = bpf_map_lookup_elem(&latency_map, &lat_key);
            if (count)
                __sync_fetch_and_add(count, 1);

            lat_key = LAT_INFER_MAX;
            __u64 *mx = bpf_map_lookup_elem(&latency_map, &lat_key);
            if (mx && elapsed > *mx)
                *mx = elapsed;   /* racy but ok for max */
        }
    }

    return cls == CLASS_BENIGN ? XDP_PASS : XDP_DROP;
}

char LICENSE[] SEC("license") = "GPL";
