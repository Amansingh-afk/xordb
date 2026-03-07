// xordb-model — download and manage ONNX models for xordb/embed.
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
	modelURL    = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
	modelName   = "all-MiniLM-L6-v2.onnx"
	modelSHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
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
  xordb-model download [--force]   Download MiniLM-L6-v2 model
  xordb-model path                 Print model file path
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

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("creating model directory: %w", err)
	}

	fmt.Printf("Downloading %s...\n", modelName)
	fmt.Printf("  From: %s\n", modelURL)
	fmt.Printf("  To:   %s\n", dest)

	// temp file mein download, phir atomic rename
	tmpFile := dest + ".download"
	if err := downloadFile(tmpFile, modelURL); err != nil {
		os.Remove(tmpFile)
		return err
	}

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

	if err := os.Rename(tmpFile, dest); err != nil {
		os.Remove(tmpFile)
		return fmt.Errorf("finalizing download: %w", err)
	}

	info, _ := os.Stat(dest)
	fmt.Printf("✓ Downloaded %s (%.1f MB)\n", modelName, float64(info.Size())/(1024*1024))
	return nil
}

const maxDownloadSize = 500 * 1024 * 1024 // 500 MB

func downloadFile(dest, url string) error {
	out, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("creating file: %w", err)
	}

	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		out.Close()
		return fmt.Errorf("HTTP request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		out.Close()
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, resp.Status)
	}

	limited := io.LimitReader(resp.Body, maxDownloadSize+1)
	var written int64
	buf := make([]byte, 32*1024)
	for {
		n, readErr := limited.Read(buf)
		if n > 0 {
			if _, writeErr := out.Write(buf[:n]); writeErr != nil {
				out.Close()
				return fmt.Errorf("writing file: %w", writeErr)
			}
			written += int64(n)
			if written > maxDownloadSize {
				out.Close()
				return fmt.Errorf("download exceeds %d MB limit", maxDownloadSize/(1024*1024))
			}
			fmt.Printf("\r  %.1f MB downloaded...", float64(written)/(1024*1024))
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			out.Close()
			return fmt.Errorf("reading response: %w", readErr)
		}
	}
	fmt.Println()

	if err := out.Close(); err != nil {
		return fmt.Errorf("closing file: %w", err)
	}
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
