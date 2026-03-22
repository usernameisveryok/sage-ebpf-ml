// SPDX-License-Identifier: GPL-2.0
/*
 * xdp_ml_v3b.c — XDP in-kernel MLP inference: MapLUT Layer 0 +
 *                standard Int32 multiplication for Layers 1 & 2.
 *
 * Architecture:  quantized MLP [12 → 64 → 64 → 4]
 * Approach:      Tail-call chain (4 stages)
 *
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║  Layer 0 — MapLUT (Map-based Lookup Table)                          ║
 * ║    Replaces ALL multiplications in Layer 0 (12→64) AND the Exit    ║
 * ║    Head (64→4) with BPF map lookups.  Zero multiply for L0.        ║
 * ╠══════════════════════════════════════════════════════════════════════╣
 * ║  Layer 1 — Standard Int32 MatMul (64→64) + ReLU                    ║
 * ║    Uses full int32 quantized weights with __s64 accumulator.        ║
 * ║    acc = Σ w1[i][j] * input[j],  result = (acc >> 16) + bias       ║
 * ╠══════════════════════════════════════════════════════════════════════╣
 * ║  Layer 2 — Standard Int32 MatMul (64→4) + argmax                   ║
 * ║    Same int32 multiply-accumulate approach for final classification. ║
 * ╠══════════════════════════════════════════════════════════════════════╣
 * ║  EarlyExit — Confidence-Based Tail-Call Bypass                     ║
 * ║    After Layer 0 + Exit Head (both via MapLUT), check top-2 logit  ║
 * ║    margin.  If confident, skip Layers 1 & 2 entirely.              ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 *
 * Program layout:
 *   Program 0  (xdp_entry)    — parse packet, extract 12 features
 *   Program 1  (xdp_layer_0)  — MapLUT Layer 0 + Exit Head + EarlyExit
 *   Program 2  (xdp_layer_1)  — Int32 MatMul (64→64) + ReLU
 *   Program 3  (xdp_classify) — Int32 MatMul (64→4) + argmax + action
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

#include "../include/model_params_v3b.h"

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

/*
 * Per-CPU input buffer for Layer 1 matmul.
 * Holds a copy of hidden[] so we can read input and write output
 * without aliasing the same map memory. This avoids using 256 bytes
 * of stack (which would exceed the 512B BPF stack limit together
 * with other local variables).
 */
struct input_buf {
    __s32 data[HIDDEN_SIZE];
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, struct input_buf);
    __uint(max_entries, 1);
} input_scratch SEC(".maps");

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
 *  MapLUT: Precomputed partial-sum lookup table (for Layer 0 + Exit)
 * ════════════════════════════════════════════════════════════════════
 *
 * BPF per-CPU array map storing precomputed partial sums.
 *
 *   Key:   j * NUM_BINS + bin_index   (j = feature index, 0..11)
 *   Value: struct lut_entry { h[64], exit_h[4] }
 *
 *   Total entries: NUM_FEATURES * NUM_BINS = 12 * 512 = 6144
 *   Per-entry size: (64 + 4) * 4 = 272 bytes
 *   Total memory per CPU: 6144 * 272 ≈ 1.6 MB
 *
 *   Populated from userspace via bpf_map_update_elem() — enables
 *   model hot-update without recompiling/reloading the eBPF program.
 */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __type(key, __u32);
    __type(value, struct lut_entry);
    __uint(max_entries, NUM_FEATURES * NUM_BINS);  /* 12 * 512 = 6144 */
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
/*  Identical to V3: zero-multiply Layer 0 computation via MapLUT.      */
/* ================================================================== */
SEC("xdp/layer0")
int xdp_layer_0(struct xdp_md *ctx)
{
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    int i, j;

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

        /* Accumulate partial sums — pure addition, zero multiply. */
        for (i = 0; i < HIDDEN_SIZE; i++)
            nctx->hidden[i] += entry->h[i];
        for (i = 0; i < NUM_CLASSES; i++)
            exit_logits[i] += entry->exit_h[i];
    }

    /* Step 3: Apply ReLU to hidden vector */
    for (i = 0; i < HIDDEN_SIZE; i++)
        nctx->hidden[i] = nctx->hidden[i] > 0 ? nctx->hidden[i] : 0;

    /*
     * ── EarlyExit Check ──────────────────────────────────────────
     *
     * The exit head logits are already computed via MapLUT above.
     * Check top-2 logit margin for confidence.
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

    if ((max1 - max2) > EARLY_EXIT_THRESHOLD) {
        nctx->logits[0]  = exit_cls;
        nctx->early_exit = 1;
        stat_inc(STAT_EARLY_EXIT);
        bpf_tail_call(ctx, &jmp_table, PROG_CLASSIFY);
    }

    /* Normal path: proceed to Layer 1 (Int32 MatMul) */
    bpf_tail_call(ctx, &jmp_table, PROG_LAYER1);
    return XDP_PASS;
}

/* ================================================================== */
/*  Program 2 — Standard Int32 MatMul Layer 1 (64→64) + ReLU          */
/*                                                                      */
/*  Uses full int32 quantized weights w1[64][64].                       */
/*  For each output neuron i:                                           */
/*    acc = Σ_j  w1[i][j] * input[j]     (__s64 accumulator)           */
/*    hidden[i] = ReLU((acc >> SCALE_FACTOR_BITS) + bias_layer_1[i])   */
/* ================================================================== */
SEC("xdp/layer1")
int xdp_layer_1(struct xdp_md *ctx)
{
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    /*
     * Copy hidden[] into a per-CPU map buffer to avoid reading and
     * writing the same map memory in the inner loop, which can
     * confuse the BPF verifier.  Using a map instead of stack avoids
     * the 512B stack limit (64 × 4 = 256 bytes).
     */
    __u32 buf_key = 0;
    struct input_buf *ibuf = bpf_map_lookup_elem(&input_scratch, &buf_key);
    if (!ibuf)
        return XDP_PASS;

    int j;
    for (j = 0; j < HIDDEN_SIZE; j++)
        ibuf->data[j] = nctx->hidden[j];

    /*
     * ── Standard Int32 MatMul (64→64) + ReLU ─────────────────────
     *
     * For each output neuron i:
     *   acc = Σ_j  w1[i][j] * input[j]   (64 multiply-accumulates)
     *   hidden[i] = ReLU((acc >> 16) + bias)
     *
     * The inner loop uses bounded iteration (j < HIDDEN_SIZE) which
     * the BPF verifier can reason about without #pragma unroll.
     */
    int i;
    for (i = 0; i < HIDDEN_SIZE; i++) {
        __s64 acc = 0;

        for (j = 0; j < HIDDEN_SIZE; j++)
            acc += (__s64)w1[i][j] * (__s64)ibuf->data[j];

        __s32 v = (__s32)(acc >> SCALE_FACTOR_BITS) + bias_layer_1[i];
        nctx->hidden[i] = v > 0 ? v : 0;   /* ReLU */
    }

    bpf_tail_call(ctx, &jmp_table, PROG_CLASSIFY);
    return XDP_PASS;
}

/* ================================================================== */
/*  Program 3 — Classification (64→4) + argmax + action                */
/*                                                                      */
/*  If early_exit == 1, use pre-stored class from the exit head.       */
/*  Otherwise, run standard int32 matmul for the final layer.           */
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
         * ── EarlyExit path ───────────────────────────────────────
         *
         * The exit head in Layer 0 (computed via MapLUT) already
         * determined the class.  Skip the final-layer computation.
         */
        cls = nctx->logits[0];

        /* Bounds-check the class to keep the verifier happy */
        if (cls < 0 || cls >= NUM_CLASSES)
            cls = 0;
    } else {
        /*
         * ── Normal path: Standard Int32 MatMul Layer 2 (64→4) ────
         *
         * For each output class i:
         *   acc = Σ_j  w2[i][j] * hidden[j]   (64 multiply-accumulates)
         *   logits[i] = (acc >> 16) + bias
         */
        int i;
        for (i = 0; i < NUM_CLASSES; i++) {
            __s64 acc = 0;
            int j;

            for (j = 0; j < HIDDEN_SIZE; j++)
                acc += (__s64)w2[i][j] * (__s64)nctx->hidden[j];

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
