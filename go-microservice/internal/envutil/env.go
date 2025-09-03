package envutil

import (
	"os"
	"strconv"
)

func Get(key, defaultValue string) string {
    if v := os.Getenv(key); v != "" { return v }
    return defaultValue
}

func GetInt(key string, defaultValue int) int {
    if v := os.Getenv(key); v != "" { if i, err := strconv.Atoi(v); err == nil { return i } }
    return defaultValue
}

func GetBool(key string, defaultValue bool) bool {
    if v := os.Getenv(key); v != "" { return v == "true" || v == "1" || v == "yes" }
    return defaultValue
}
