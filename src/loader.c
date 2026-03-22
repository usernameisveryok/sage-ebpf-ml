// SPDX-License-Identifier: GPL-2.0-or-later
/*
 * loader.c — Userspace loader/controller for the XDP ML classifier.
 *
 * Loads xdp_ml.o, populates the tail-call jump table, attaches the
 * entry XDP program to a network interface, and monitors stats.
 *
 * The eBPF object contains four XDP programs chained via tail calls:
 *   0: xdp_entry   — packet parse + flow tracking + feature extraction
 *   1: xdp_layer_0 — input standardisation + layer 0 (6→32) + ReLU
 *   2: xdp_layer_1 — layer 1 (32→32) + ReLU
 *   3: xdp_layer_2 — layer 2 (32→4) + argmax + classify
 *
 * Build:
 *   cc -O2 -Wall -o loader loader.c -lbpf -lelf -lz
 *
 * Usage:
 *   sudo ./loader <interface> [options]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <getopt.h>
#include <net/if.h>
#include <linux/if_link.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */
#define DEFAULT_BPF_OBJ   "xdp_ml.o"
#define POLL_INTERVAL_SEC  2

/* Stats indices — must match the kernel-side defines */
enum stat_idx {
    STAT_TOTAL_PKTS   = 0,
    STAT_INFER_RUNS   = 1,
    STAT_CLASS_BENIGN = 2,
    STAT_CLASS_DDOS   = 3,
    STAT_CLASS_PORTSCAN = 4,
    STAT_CLASS_BRUTE  = 5,
    STAT_BLOCKED      = 6,
    STAT_MAX          = 8,
};

static const char *stat_labels[STAT_MAX] = {
    [STAT_TOTAL_PKTS]   = "Total packets",
    [STAT_INFER_RUNS]   = "Inference runs",
    [STAT_CLASS_BENIGN]  = "BENIGN",
    [STAT_CLASS_DDOS]    = "DDOS",
    [STAT_CLASS_PORTSCAN]= "PORTSCAN",
    [STAT_CLASS_BRUTE]   = "BRUTEFORCE",
    [STAT_BLOCKED]       = "Blocked pkts",
    [7]                  = NULL,
};

/* Tail-call program indices — must match the kernel-side defines */
#define PROG_ENTRY   0
#define PROG_LAYER0  1
#define PROG_LAYER1  2
#define PROG_LAYER2  3
#define PROG_MAX     4

/* Program section names corresponding to each tail-call index */
static const char *prog_names[PROG_MAX] = {
    [PROG_ENTRY]  = "xdp_entry",
    [PROG_LAYER0] = "xdp_layer_0",
    [PROG_LAYER1] = "xdp_layer_1",
    [PROG_LAYER2] = "xdp_layer_2",
};

/* ------------------------------------------------------------------ */
/* Globals                                                             */
/* ------------------------------------------------------------------ */
static volatile sig_atomic_t g_running = 1;
static int   g_ifindex;
static __u32 g_xdp_flags;

static void sig_handler(int sig) { (void)sig; g_running = 0; }

static int install_signal_handlers(void)
{
    struct sigaction sa = {};
    sa.sa_handler = sig_handler;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGINT, &sa, NULL) < 0 ||
        sigaction(SIGTERM, &sa, NULL) < 0) {
        perror("sigaction");
        return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Usage                                                               */
/* ------------------------------------------------------------------ */
static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s <interface> [options]\n\n"
        "Options:\n"
        "  -S    SKB (generic) XDP mode\n"
        "  -N    Native XDP mode (default)\n"
        "  -u    Unload/detach only\n"
        "  -v    Verbose libbpf output\n", prog);
    exit(EXIT_FAILURE);
}

/* ------------------------------------------------------------------ */
/* Stats                                                               */
/* ------------------------------------------------------------------ */
static int read_stats(int map_fd, __u64 out[STAT_MAX])
{
    int ncpus = libbpf_num_possible_cpus();
    if (ncpus < 0) return -1;

    __u64 *values = calloc(ncpus, sizeof(__u64));
    if (!values) return -1;

    for (int key = 0; key < STAT_MAX; key++) {
        memset(values, 0, ncpus * sizeof(__u64));
        out[key] = 0;
        if (bpf_map_lookup_elem(map_fd, &key, values) == 0)
            for (int c = 0; c < ncpus; c++)
                out[key] += values[c];
    }
    free(values);
    return 0;
}

static void print_stats(const __u64 st[STAT_MAX])
{
    char buf[64];
    time_t now = time(NULL);
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
    printf("[%s] Stats:\n", buf);
    for (int i = 0; i < STAT_MAX; i++) {
        if (!stat_labels[i]) continue;
        printf("  %-20s %llu\n", stat_labels[i],
               (unsigned long long)st[i]);
    }
    printf("\n");
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* XDP attach / detach                                                 */
/* ------------------------------------------------------------------ */
static void detach_xdp(int ifindex, __u32 flags)
{
    int err = bpf_set_link_xdp_fd(ifindex, -1, flags);
    if (err)
        fprintf(stderr, "WARN: detach failed: %s\n", strerror(-err));
    else
        printf("XDP program detached from ifindex %d\n", ifindex);
}

/* ------------------------------------------------------------------ */
/* libbpf print callbacks                                              */
/* ------------------------------------------------------------------ */
static int verbose_print(enum libbpf_print_level l,
                         const char *fmt, va_list ap)
{ (void)l; return vfprintf(stderr, fmt, ap); }

static int quiet_print(enum libbpf_print_level l,
                       const char *fmt, va_list ap)
{ (void)l; (void)fmt; (void)ap; return 0; }

/* ================================================================== */
/* main                                                                */
/* ================================================================== */
int main(int argc, char **argv)
{
    const char *ifname = NULL;
    int  unload_only = 0, verbose = 0, mode_set = 0;
    __u32 xdp_flags = XDP_FLAGS_UPDATE_IF_NOEXIST;

    if (argc < 2) usage(argv[0]);
    ifname = argv[1];

    optind = 2;
    int opt;
    while ((opt = getopt(argc, argv, "SNuv")) != -1) {
        switch (opt) {
        case 'S': xdp_flags |= XDP_FLAGS_SKB_MODE; mode_set = 1; break;
        case 'N': xdp_flags |= XDP_FLAGS_DRV_MODE; mode_set = 1; break;
        case 'u': unload_only = 1; break;
        case 'v': verbose = 1; break;
        default:  usage(argv[0]);
        }
    }
    if (!mode_set)
        xdp_flags |= XDP_FLAGS_DRV_MODE;

    unsigned int ifindex = if_nametoindex(ifname);
    if (!ifindex) {
        fprintf(stderr, "ERROR: interface '%s' not found\n", ifname);
        return EXIT_FAILURE;
    }
    g_ifindex = ifindex;
    g_xdp_flags = xdp_flags;

    /* ---- Unload only ---------------------------------------------- */
    if (unload_only) {
        detach_xdp(ifindex, xdp_flags);
        return EXIT_SUCCESS;
    }

    /* ---- Setup ---------------------------------------------------- */
    libbpf_set_print(verbose ? verbose_print : quiet_print);
    if (install_signal_handlers()) return EXIT_FAILURE;

    /* ---- Open BPF object ------------------------------------------ */
    struct bpf_object *obj = bpf_object__open_file(DEFAULT_BPF_OBJ, NULL);
    if (!obj) {
        fprintf(stderr, "ERROR: open '%s': %s\n",
                DEFAULT_BPF_OBJ, strerror(errno));
        return EXIT_FAILURE;
    }

    /* ---- Load into kernel ----------------------------------------- */
    int err = bpf_object__load(obj);
    if (err) {
        fprintf(stderr, "ERROR: load BPF object: %s\n", strerror(-err));
        bpf_object__close(obj);
        return EXIT_FAILURE;
    }

    /* ---- Populate tail-call jump table ---------------------------- */
    struct bpf_map *jmp_map =
        bpf_object__find_map_by_name(obj, "jmp_table");
    if (!jmp_map) {
        fprintf(stderr, "ERROR: 'jmp_table' map not found\n");
        bpf_object__close(obj);
        return EXIT_FAILURE;
    }
    int jmp_fd = bpf_map__fd(jmp_map);

    for (int i = 0; i < PROG_MAX; i++) {
        struct bpf_program *p =
            bpf_object__find_program_by_name(obj, prog_names[i]);
        if (!p) {
            fprintf(stderr, "ERROR: program '%s' not found\n",
                    prog_names[i]);
            bpf_object__close(obj);
            return EXIT_FAILURE;
        }
        int fd = bpf_program__fd(p);
        __u32 key = i;
        if (bpf_map_update_elem(jmp_fd, &key, &fd, BPF_ANY)) {
            fprintf(stderr, "ERROR: failed to insert prog[%d] '%s' "
                    "into jmp_table: %s\n", i, prog_names[i],
                    strerror(errno));
            bpf_object__close(obj);
            return EXIT_FAILURE;
        }
        if (verbose)
            fprintf(stderr, "INFO: jmp_table[%d] = %s (fd %d)\n",
                    i, prog_names[i], fd);
    }

    /* ---- Find stats map ------------------------------------------- */
    struct bpf_map *smap =
        bpf_object__find_map_by_name(obj, "stats_map");
    if (!smap) {
        fprintf(stderr, "ERROR: 'stats_map' not found\n");
        bpf_object__close(obj);
        return EXIT_FAILURE;
    }
    int stats_fd = bpf_map__fd(smap);

    /* ---- Find latency map ----------------------------------------- */
    struct bpf_map *lmap =
        bpf_object__find_map_by_name(obj, "latency_map");
    if (!lmap) {
        fprintf(stderr, "WARN: 'latency_map' not found, latency stats disabled\n");
    }
    int latency_fd = lmap ? bpf_map__fd(lmap) : -1;

    /* ---- Find entry program and attach to interface --------------- */
    struct bpf_program *entry =
        bpf_object__find_program_by_name(obj, "xdp_entry");
    if (!entry) {
        fprintf(stderr, "ERROR: 'xdp_entry' program not found\n");
        bpf_object__close(obj);
        return EXIT_FAILURE;
    }
    int entry_fd = bpf_program__fd(entry);

    err = bpf_set_link_xdp_fd(ifindex, entry_fd, xdp_flags);
    if (err) {
        fprintf(stderr,
            "ERROR: attach XDP to %s: %s\n"
            "HINT: try -S for SKB mode, or -u to detach first\n",
            ifname, strerror(-err));
        bpf_object__close(obj);
        return EXIT_FAILURE;
    }

    printf("XDP ML classifier loaded on %s (ifindex %u, %s mode)\n"
           "  Model: MLP [6 → 32 → 32 → 4], int32 quantized\n"
           "  Programs: entry + 3 tail-call stages\n"
           "  Press Ctrl+C to stop.\n\n",
           ifname, ifindex,
           (xdp_flags & XDP_FLAGS_SKB_MODE) ? "SKB" : "native");

    /* ---- Monitor loop --------------------------------------------- */
    __u64 stats[STAT_MAX];
    while (g_running) {
        sleep(POLL_INTERVAL_SEC);
        if (!g_running) break;
        if (read_stats(stats_fd, stats) == 0) {
            print_stats(stats);
            /* Read latency stats */
            if (latency_fd >= 0) {
                int ncpus = libbpf_num_possible_cpus();
                if (ncpus > 0) {
                    __u64 *pcpu = calloc(ncpus, sizeof(__u64));
                    if (pcpu) {
                        __u64 lat_values[4] = {0};
                        for (int i = 0; i < 4; i++) {
                            __u32 lat_key = i;
                            memset(pcpu, 0, ncpus * sizeof(__u64));
                            if (bpf_map_lookup_elem(latency_fd, &lat_key, pcpu) == 0) {
                                for (int c = 0; c < ncpus; c++)
                                    lat_values[i] += pcpu[c];
                            }
                        }
                        if (lat_values[1] > 0) {
                            printf("  Avg inference: %llu ns, Max: %llu ns (%llu runs)\n",
                                   (unsigned long long)(lat_values[0] / lat_values[1]),
                                   (unsigned long long)lat_values[2],
                                   (unsigned long long)lat_values[1]);
                        }
                        free(pcpu);
                    }
                }
            }
        }
    }

    /* ---- Cleanup -------------------------------------------------- */
    printf("\nShutting down...\n");
    detach_xdp(ifindex, xdp_flags);
    bpf_object__close(obj);
    return EXIT_SUCCESS;
}
