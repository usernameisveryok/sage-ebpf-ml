#!/usr/bin/env python3
"""test_inference.py — Verify integer-only MLP inference."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
import numpy as np

from train_model import (
    generate_dataset, quantize_model, verify_quantized,
    MLP, SCALE_BITS, SCALE, NUM_FEATURES, NUM_CLASSES, HIDDEN_SIZE,
    CLASS_NAMES, generate_class_samples,
)

SEED = 42
PASS_THRESHOLD = 0.95  # minimum acceptable accuracy


def build_trained_model():
    """Generate data, train model, return everything needed for testing."""
    np.random.seed(SEED)
    X, y = generate_dataset()
    split = int(0.8 * len(y))
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1e-8

    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    layer_sizes = [NUM_FEATURES, HIDDEN_SIZE, HIDDEN_SIZE, NUM_CLASSES]
    from train_model import train_model
    model = train_model(X_train_n, y_train, X_test_n, y_test,
                        layer_sizes, epochs=80, batch_size=256, lr=0.05)
    W_q, b_q, norm_s, norm_o = quantize_model(model, mean, std)
    return model, W_q, b_q, norm_s, norm_o, X_test, y_test, mean, std


def test_basic_accuracy(W_q, b_q, norm_s, norm_o, X_test, y_test):
    """Integer-only accuracy must exceed PASS_THRESHOLD."""
    acc, _ = verify_quantized(None, W_q, b_q, norm_s, norm_o, X_test, y_test)
    ok = acc >= PASS_THRESHOLD
    print(f"  [{'PASS' if ok else 'FAIL'}] Basic accuracy: {acc:.4f} "
          f"(threshold {PASS_THRESHOLD})")
    return ok


def test_float_vs_int(model, W_q, b_q, norm_s, norm_o,
                      X_test, y_test, mean, std):
    """Accuracy delta between float and int must be < 2%."""
    X_test_n = (X_test - mean) / std
    float_pred = model.predict(X_test_n)
    float_acc = (float_pred == y_test).mean()
    int_acc, _ = verify_quantized(None, W_q, b_q, norm_s, norm_o, X_test, y_test)
    delta = abs(float_acc - int_acc)
    ok = delta < 0.02
    print(f"  [{'PASS' if ok else 'FAIL'}] Float/Int delta: {delta:.4f} "
          f"(float={float_acc:.4f}, int={int_acc:.4f})")
    return ok


def test_per_class(W_q, b_q, norm_s, norm_o, X_test, y_test):
    """Each class must achieve > 90% recall."""
    _, preds = verify_quantized(None, W_q, b_q, norm_s, norm_o, X_test, y_test)
    all_ok = True
    for c in range(NUM_CLASSES):
        mask = y_test == c
        if mask.sum() == 0:
            continue
        recall = (preds[mask] == c).mean()
        ok = recall > 0.90
        if not ok:
            all_ok = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {CLASS_NAMES[c]:>12s} recall: "
              f"{recall:.4f}")
    return all_ok


def test_edge_large_features(W_q, b_q, norm_s, norm_o):
    """Very large feature values should not crash and must return valid class."""
    X_big = np.array([[1500, 60, 65535, 1_000_000, 10000, 1500]], dtype=np.float64)
    y_dummy = np.array([0])
    _, preds = verify_quantized(None, W_q, b_q, norm_s, norm_o, X_big, y_dummy)
    ok = 0 <= preds[0] < NUM_CLASSES
    print(f"  [{'PASS' if ok else 'FAIL'}] Large features → class {preds[0]}")
    return ok


def test_edge_small_features(W_q, b_q, norm_s, norm_o):
    """Very small / zero feature values should return valid class."""
    X_small = np.array([[64, 20, 0, 0, 1, 64]], dtype=np.float64)
    y_dummy = np.array([0])
    _, preds = verify_quantized(None, W_q, b_q, norm_s, norm_o, X_small, y_dummy)
    ok = 0 <= preds[0] < NUM_CLASSES
    print(f"  [{'PASS' if ok else 'FAIL'}] Small features → class {preds[0]}")
    return ok


def test_edge_uniform_features(W_q, b_q, norm_s, norm_o):
    """All features identical — model should still return a valid class."""
    for val in [0, 500, 65535]:
        X_uni = np.full((1, NUM_FEATURES), val, dtype=np.float64)
        y_dummy = np.array([0])
        _, preds = verify_quantized(None, W_q, b_q, norm_s, norm_o, X_uni, y_dummy)
        ok = 0 <= preds[0] < NUM_CLASSES
        print(f"  [{'PASS' if ok else 'FAIL'}] Uniform({val}) → class {preds[0]}")
        if not ok:
            return False
    return True


def main():
    print("=" * 60)
    print("  Integer Inference Test Suite")
    print("=" * 60)

    print("\n[1/2] Training model (this may take a moment)...")
    model, W_q, b_q, ns, no, X_test, y_test, mean, std = build_trained_model()

    print("\n[2/2] Running tests...\n")
    results = []
    results.append(("basic_accuracy",    test_basic_accuracy(W_q, b_q, ns, no, X_test, y_test)))
    results.append(("float_vs_int",      test_float_vs_int(model, W_q, b_q, ns, no, X_test, y_test, mean, std)))
    results.append(("per_class_recall",  test_per_class(W_q, b_q, ns, no, X_test, y_test)))
    results.append(("edge_large",        test_edge_large_features(W_q, b_q, ns, no)))
    results.append(("edge_small",        test_edge_small_features(W_q, b_q, ns, no)))
    results.append(("edge_uniform",      test_edge_uniform_features(W_q, b_q, ns, no)))

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed}/{total} tests passed")
    if passed == total:
        print("  *** ALL TESTS PASSED ***")
    else:
        for name, ok in results:
            if not ok:
                print(f"  FAILED: {name}")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
