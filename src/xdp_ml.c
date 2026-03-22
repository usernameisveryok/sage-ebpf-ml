// SPDX-License-Identifier: GPL-2.0
/*
 * xdp_ml.c — XDP in-kernel MLP inference for packet classification
 *
 * Architecture:  quantized MLP [6 → 32 → 32 → 4]
 * Approach:      Tail-call chain to stay within 512-byte stack limit.
 *
 *   Program 0  (xdp_entry)    — parse pkt, update flow, trigger inference
 *   Program 1  (xdp_layer_0)  — input standardisation + layer 0 + ReLU
 *   Program 2  (xdp_layer_1)  — layer 1 + ReLU
 *   Program 3  (xdp_layer_2)  — layer 2 + argmax + classify
 *
 * Intermediate hidden-layer vectors are stored in a per-CPU map.
 */

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

#include "../include/model_params.h"

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

/* Tail-call program indices */
#define PROG_ENTRY    0
#define PROG_LAYER0   1
#define PROG_LAYER1   2
#define PROG_LAYER2   3
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

struct flow_features {
    __s64 pkt_len_sum;
    __s64 hdr_len_sum;
    __s32 dst_port;
    __s64 fwd_iat_sum;
    __s64 last_seen;
    __s32 total_fwd_pkts;
    __s32 fwd_pkt_len_max;
    __s32 pkt_count;
    __s32 classified;
    __s32 class_result;
};

/*
 * Per-CPU scratch space shared between tail-call stages.
 * Contains features, hidden vectors, and the flow key for the
 * current packet being processed.
 */
struct nn_ctx {
    __s32 features[NUM_FEATURES];   /* 6 × 4 =  24 bytes */
    __s32 hidden[HIDDEN_SIZE];      /* 32 × 4 = 128 bytes */
    __s32 logits[NUM_CLASSES];      /* 4 × 4 =  16 bytes */
    struct flow_key key;            /* 16 bytes */
    struct flow_key rkey;           /* 16 bytes */
    __s32 is_fin_rst;               /* 4 bytes  */
};                                  /* total ≈ 204 bytes */

/* ------------------------------------------------------------------ */
/*  BPF Maps                                                           */
/* ------------------------------------------------------------------ */
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, struct flow_key);
    __type(value, struct flow_features);
    __uint(max_entries, MAX_FLOWS);
} flow_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, struct flow_key);
    __type(value, __u32);
    __uint(max_entries, MAX_BLOCK);
} blocklist_map SEC(".maps");

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
#define LAT_INFER_COUNT   1   /* number of inference runs */
#define LAT_INFER_MAX     2   /* max single inference time */
#define LAT_START         3   /* temp: start timestamp */

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */
static __always_inline void stat_inc(__u32 idx)
{
    __u64 *val = bpf_map_lookup_elem(&stats_map, &idx);
    if (val)
        __sync_fetch_and_add(val, 1);
}

static __always_inline __s64 safe_div(__s64 a, __s64 b)
{
    if (b == 0) return 0;
    __u64 ua = a < 0 ? (__u64)(-a) : (__u64)a;
    __u64 ub = b < 0 ? (__u64)(-b) : (__u64)b;
    __s64 r = (__s64)(ua / ub);
    return ((a ^ b) < 0) ? -r : r;
}

static __always_inline struct nn_ctx *get_ctx(void)
{
    __u32 key = 0;
    return bpf_map_lookup_elem(&nn_scratch, &key);
}

/* ================================================================== */
/*  Program 0 — Packet parse, flow tracking, inference trigger         */
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

    if (proto == IPPROTO_TCP) {
        struct tcphdr *tcph = (void *)iph + ip_hdr_len;
        if ((void *)(tcph + 1) > data_end)
            return XDP_PASS;
        src_port   = bpf_ntohs(tcph->source);
        dst_port   = bpf_ntohs(tcph->dest);
        l4_hdr_len = tcph->doff * 4;
        is_fin     = tcph->fin;
        is_rst     = tcph->rst;
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
    key.src_ip = src_ip;  key.dst_ip = dst_ip;
    key.src_port = src_port;  key.dst_port = dst_port;
    key.protocol = proto;

    /* ---- Blocklist check ------------------------------------------ */
    if (bpf_map_lookup_elem(&blocklist_map, &key)) {
        stat_inc(STAT_BLOCKED);
        return XDP_DROP;
    }
    struct flow_key rkey = {};
    rkey.src_ip = dst_ip;  rkey.dst_ip = src_ip;
    rkey.src_port = dst_port;  rkey.dst_port = src_port;
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
        newf.dst_port        = (__s32)dst_port;
        newf.last_seen       = (__s64)now;
        newf.total_fwd_pkts  = 1;
        newf.fwd_pkt_len_max = (__s32)total_pkt_len;
        newf.pkt_count       = 1;
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
    if (flow->last_seen > 0) {
        __s64 iat = (__s64)now - flow->last_seen;
        if (iat > 0)
            flow->fwd_iat_sum += iat;
    }
    flow->last_seen = (__s64)now;
    flow->pkt_count++;

    /* ---- Trigger? ------------------------------------------------- */
    if (!is_fin && !is_rst && flow->pkt_count < PKT_THRESHOLD)
        return XDP_PASS;

    /* ---- Compute features & store in scratch ---------------------- */
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    __s32 cnt = flow->pkt_count > 0 ? flow->pkt_count : 1;
    __s32 iat_div = cnt > 1 ? cnt - 1 : 1;

    nctx->features[0] = (__s32)safe_div(flow->pkt_len_sum, (__s64)cnt);
    nctx->features[1] = (__s32)safe_div(flow->hdr_len_sum, (__s64)cnt);
    nctx->features[2] = flow->dst_port;
    nctx->features[3] = (__s32)safe_div(
                            safe_div(flow->fwd_iat_sum, (__s64)iat_div),
                            1000LL);
    nctx->features[4] = flow->total_fwd_pkts;
    nctx->features[5] = flow->fwd_pkt_len_max;

    nctx->key = key;
    nctx->rkey = rkey;
    nctx->is_fin_rst = (is_fin || is_rst) ? 1 : 0;

    /* Record inference start time */
    {
        __u32 lat_key = LAT_START;
        __u64 start = bpf_ktime_get_ns();
        bpf_map_update_elem(&latency_map, &lat_key, &start, BPF_ANY);
    }

    /* Jump to layer 0 */
    bpf_tail_call(ctx, &jmp_table, PROG_LAYER0);

    /* If tail call fails, just pass */
    return XDP_PASS;
}

/* ================================================================== */
/*  Program 1 — Standardise input + Layer 0 (6 → 32) + ReLU           */
/* ================================================================== */
SEC("xdp/layer0")
int xdp_layer_0(struct xdp_md *ctx)
{
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    /* Standardise: x_norm = norm_scale * x - norm_offset */
    __s32 x_norm[NUM_FEATURES];
    int j;
    for (j = 0; j < NUM_FEATURES; j++) {
        x_norm[j] = (__s32)((__s64)norm_scale[j] * (__s64)nctx->features[j]
                            - (__s64)norm_offset[j]);
    }

    /* Layer 0: x_norm (6) → hidden (32) + ReLU */
    int i;
    for (i = 0; i < HIDDEN_SIZE; i++) {
        __s64 sum = 0;
        for (j = 0; j < NUM_FEATURES; j++)
            sum += (__s64)weight_layer_0[i][j] * (__s64)x_norm[j];
        __s32 v = (__s32)(sum >> SCALE_FACTOR_BITS) + bias_layer_0[i];
        nctx->hidden[i] = v > 0 ? v : 0;
    }

    bpf_tail_call(ctx, &jmp_table, PROG_LAYER1);
    return XDP_PASS;
}

/* ================================================================== */
/*  Program 2 — Layer 1 (32 → 32) + ReLU                              */
/* ================================================================== */
SEC("xdp/layer1")
int xdp_layer_1(struct xdp_md *ctx)
{
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    /*
     * Read hidden[] into a local temp, then write back.
     * This avoids reading & writing the same map memory in
     * the inner loop, which can confuse the verifier.
     */
    __s32 input[HIDDEN_SIZE];
    int j;
    for (j = 0; j < HIDDEN_SIZE; j++)
        input[j] = nctx->hidden[j];

    int i;
    for (i = 0; i < HIDDEN_SIZE; i++) {
        __s64 sum = 0;
        for (j = 0; j < HIDDEN_SIZE; j++)
            sum += (__s64)weight_layer_1[i][j] * (__s64)input[j];
        __s32 v = (__s32)(sum >> SCALE_FACTOR_BITS) + bias_layer_1[i];
        nctx->hidden[i] = v > 0 ? v : 0;
    }

    bpf_tail_call(ctx, &jmp_table, PROG_LAYER2);
    return XDP_PASS;
}

/* ================================================================== */
/*  Program 3 — Layer 2 (32 → 4) + argmax + classify                  */
/* ================================================================== */
SEC("xdp/layer2")
int xdp_layer_2(struct xdp_md *ctx)
{
    struct nn_ctx *nctx = get_ctx();
    if (!nctx)
        return XDP_PASS;

    stat_inc(STAT_INFER_RUNS);

    /* Layer 2: hidden (32) → logits (4) */
    int i, j;
    for (i = 0; i < NUM_CLASSES; i++) {
        __s64 sum = 0;
        for (j = 0; j < HIDDEN_SIZE; j++)
            sum += (__s64)weight_layer_2[i][j] * (__s64)nctx->hidden[j];
        nctx->logits[i] = (__s32)(sum >> SCALE_FACTOR_BITS) + bias_layer_2[i];
    }

    /* Argmax */
    __s32 cls = 0;
    for (i = 1; i < NUM_CLASSES; i++)
        if (nctx->logits[i] > nctx->logits[cls])
            cls = i;

    /* Update flow state */
    struct flow_features *flow = bpf_map_lookup_elem(&flow_map, &nctx->key);
    if (flow) {
        flow->classified   = 1;
        flow->class_result = cls;
    }

    /* Stats */
    __u32 cls_stat;
    switch (cls) {
    case CLASS_BENIGN:     cls_stat = STAT_CLASS_BENIGN;   break;
    case CLASS_DDOS:       cls_stat = STAT_CLASS_DDOS;     break;
    case CLASS_PORTSCAN:   cls_stat = STAT_CLASS_PORTSCAN; break;
    case CLASS_BRUTEFORCE: cls_stat = STAT_CLASS_BRUTE;    break;
    default:               cls_stat = STAT_CLASS_BENIGN;   break;
    }
    stat_inc(cls_stat);

    /* Blocklist malicious flows */
    if (cls != CLASS_BENIGN) {
        __u32 label = (__u32)cls;
        bpf_map_update_elem(&blocklist_map, &nctx->key,  &label, BPF_ANY);
        bpf_map_update_elem(&blocklist_map, &nctx->rkey, &label, BPF_ANY);
    }

    /* Cleanup finished flows */
    if (nctx->is_fin_rst)
        bpf_map_delete_elem(&flow_map, &nctx->key);

    /* Record inference latency */
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
                *mx = elapsed;  /* racy but ok for max */
        }
    }

    return cls == CLASS_BENIGN ? XDP_PASS : XDP_DROP;
}

char LICENSE[] SEC("license") = "GPL";
