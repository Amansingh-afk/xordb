// Command xordb-model manages ONNX model files for xordb/embed.
//
// Usage:
//
//	xordb-model download          Download the default MiniLM model
//	xordb-model download --force  Re-download even if already present
//	xordb-model path              Print the model file path
//	xordb-model info              Print model info and status
package main

import (
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"xordb/embed"
)

const (
	modelURL  = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
	modelName = "all-MiniLM-L6-v2.onnx"
	// SHA-256 of the FP32 ONNX model from HuggingFace (for integrity verification).
	// Set to empty to skip verification (useful during development).
	modelSHA256 = ""
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "download":
		force := len(os.Args) > 2 && os.Args[2] == "--force"
		if err := downloadModel(force); err != nil {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			os.Exit(1)
		}
	case "path":
		printModelPath()
	case "info":
		printModelInfo()
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`xordb-model — manage ONNX models for xordb/embed

Usage:
  xordb-model download [--force]   Download the default MiniLM-L6-v2 model
  xordb-model path                 Print the expected model file path
  xordb-model info                 Print model info and status
  xordb-model help                 Show this help

Environment:
  XORDB_MODEL_PATH    Override model file location
  XDG_DATA_HOME       Override data directory (default: ~/.local/share)`)
}

func downloadModel(force bool) error {
	dir := embed.ModelDir()
	dest := filepath.Join(dir, modelName)

	if !force {
		if _, err := os.Stat(dest); err == nil {
			fmt.Printf("✓ Model already exists at %s\n", dest)
			fmt.Println("  Use --force to re-download.")
			return nil
		}
	}

	// Create directory.
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating model directory: %w", err)
	}

	fmt.Printf("Downloading %s...\n", modelName)
	fmt.Printf("  From: %s\n", modelURL)
	fmt.Printf("  To:   %s\n", dest)

	// Download to a temp file first, then rename for atomicity.
	tmpFile := dest + ".download"
	if err := downloadFile(tmpFile, modelURL); err != nil {
		os.Remove(tmpFile)
		return err
	}

	// Verify SHA-256 if configured.
	if modelSHA256 != "" {
		hash, err := fileSHA256(tmpFile)
		if err != nil {
			os.Remove(tmpFile)
			return fmt.Errorf("computing checksum: %w", err)
		}
		if !strings.EqualFold(hash, modelSHA256) {
			os.Remove(tmpFile)
			return fmt.Errorf("checksum mismatch: got %s, want %s", hash, modelSHA256)
		}
		fmt.Println("  ✓ SHA-256 verified")
	}

	// Atomic rename.
	if err := os.Rename(tmpFile, dest); err != nil {
		os.Remove(tmpFile)
		return fmt.Errorf("finalizing download: %w", err)
	}

	info, _ := os.Stat(dest)
	fmt.Printf("✓ Downloaded %s (%.1f MB)\n", modelName, float64(info.Size())/(1024*1024))
	return nil
}

func downloadFile(dest, url string) error {
	out, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("creating file: %w", err)
	}
	defer out.Close()

	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return fmt.Errorf("HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	// Show download progress.
	var written int64
	buf := make([]byte, 32*1024)
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := out.Write(buf[:n]); writeErr != nil {
				return fmt.Errorf("writing file: %w", writeErr)
			}
			written += int64(n)
			fmt.Printf("\r  %.1f MB downloaded...", float64(written)/(1024*1024))
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fmt.Errorf("reading response: %w", readErr)
		}
	}
	fmt.Println()

	return nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func printModelPath() {
	path, err := embed.DefaultModelPath()
	if err != nil {
		// Print the expected path even if the file doesn't exist.
		fmt.Println(filepath.Join(embed.ModelDir(), modelName))
		return
	}
	fmt.Println(path)
}

func printModelInfo() {
	fmt.Println("Model: all-MiniLM-L6-v2 (sentence-transformers)")
	fmt.Println("Format: ONNX (FP32)")
	fmt.Println("Embedding dims: 384")
	fmt.Println("Max sequence length: 256 tokens")
	fmt.Println("License: Apache 2.0")
	fmt.Println()

	dir := embed.ModelDir()
	dest := filepath.Join(dir, modelName)

	if info, err := os.Stat(dest); err == nil {
		fmt.Printf("Status: ✓ Downloaded\n")
		fmt.Printf("Path: %s\n", dest)
		fmt.Printf("Size: %.1f MB\n", float64(info.Size())/(1024*1024))
	} else {
		fmt.Printf("Status: ✗ Not downloaded\n")
		fmt.Printf("Expected path: %s\n", dest)
		fmt.Println("\nRun 'xordb-model download' to download the model.")
	}
}
