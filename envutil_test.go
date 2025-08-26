package main

import (
	"os"
	"testing"
	"time"
)

func TestGetEnvHelpers(t *testing.T) {
    os.Setenv("TEST_STR", "value")
    os.Setenv("TEST_INT", "42")
    os.Setenv("TEST_FLOAT", "3.14")
    os.Setenv("TEST_BOOL_T", "true")
    os.Setenv("TEST_BOOL_F", "0")
    os.Setenv("TEST_DUR", "250ms")

    if v := getEnv("TEST_STR", "x"); v != "value" { t.Fatalf("expected value got %s", v) }
    if v := getEnvInt("TEST_INT", 1); v != 42 { t.Fatalf("expected 42 got %d", v) }
    if v := getEnvFloat("TEST_FLOAT", 1); v < 3.139 || v > 3.141 { t.Fatalf("expected ~3.14 got %f", v) }
    if v := getEnvBool("TEST_BOOL_T", false); v != true { t.Fatalf("expected true") }
    if v := getEnvBool("TEST_BOOL_F", true); v != false { t.Fatalf("expected false") }
    if v := getEnvDuration("TEST_DUR", time.Second); v != 250*time.Millisecond { t.Fatalf("expected 250ms got %v", v) }
}
