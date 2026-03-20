package cache

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash/crc32"
	"io"
	"time"

	"github.com/Amansingh-afk/xordb/hdc"
)

const (
	headerSize    = 32
	formatMagic   = "XRDB"
	formatVersion = 2

	maxKeyLen     = 1 << 20  // 1 MB
	maxValLen     = 1 << 24  // 16 MB
	maxEntryCount = 1 << 24  // ~16M entries
	maxPayloadLen = 1 << 32  // 4 GB hard cap on payload read
)

// EncodeSnapshot writes a binary-encoded snapshot to w.
// Format: 32-byte header + entry payload.
// Values are JSON-encoded. Vectors are raw uint64 bytes (little-endian).
func EncodeSnapshot(w io.Writer, s Snapshot) error {
	// Encode entries into a buffer first to compute CRC.
	var payload bytes.Buffer
	for _, e := range s.Entries {
		if err := encodeEntry(&payload, e, s.Dims); err != nil {
			return err
		}
	}

	payloadBytes := payload.Bytes()
	crc := crc32.ChecksumIEEE(payloadBytes)

	// Write header.
	var hdr [headerSize]byte
	copy(hdr[0:4], formatMagic)
	binary.LittleEndian.PutUint16(hdr[4:6], formatVersion)
	binary.LittleEndian.PutUint16(hdr[6:8], 0) // flags reserved
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(s.Dims))
	binary.LittleEndian.PutUint32(hdr[12:16], uint32(s.Capacity))
	binary.LittleEndian.PutUint32(hdr[16:20], uint32(len(s.Entries)))
	binary.LittleEndian.PutUint32(hdr[20:24], crc)
	// hdr[24:32] reserved, zero

	if _, err := w.Write(hdr[:]); err != nil {
		return err
	}
	_, err := w.Write(payloadBytes)
	return err
}

// DecodeSnapshot reads a binary-encoded snapshot.
// dims is the expected vector dimensionality — mismatches are rejected.
func DecodeSnapshot(r io.Reader, dims int) (Snapshot, error) {
	var hdr [headerSize]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		return Snapshot{}, fmt.Errorf("cache: read header: %w", err)
	}

	if string(hdr[0:4]) != formatMagic {
		return Snapshot{}, fmt.Errorf("cache: invalid magic %q (want %q)", hdr[0:4], formatMagic)
	}
	version := binary.LittleEndian.Uint16(hdr[4:6])
	if version != formatVersion {
		return Snapshot{}, fmt.Errorf("cache: format version %d unsupported (want %d)", version, formatVersion)
	}

	fileDims := int(binary.LittleEndian.Uint32(hdr[8:12]))
	if fileDims != dims {
		return Snapshot{}, fmt.Errorf("cache: file dims %d does not match cache dims %d", fileDims, dims)
	}

	capacity := int(binary.LittleEndian.Uint32(hdr[12:16]))
	count := int(binary.LittleEndian.Uint32(hdr[16:20]))
	expectedCRC := binary.LittleEndian.Uint32(hdr[20:24])

	if count < 0 || count > maxEntryCount {
		return Snapshot{}, fmt.Errorf("cache: entry count %d out of range (max %d)", count, maxEntryCount)
	}

	// Compute upper bound on payload size to prevent unbounded reads.
	// Use realistic per-entry sizes rather than maximum key/value lengths,
	// which would make the limit effectively useless.
	nw := hdc.NumWords(dims)
	entryOverhead := int64(4 + 4096 + int64(nw)*8 + 16 + 4 + 1<<20)
	maxPayload := int64(count) * entryOverhead
	if maxPayload > maxPayloadLen {
		maxPayload = maxPayloadLen
	}

	payloadBytes, err := io.ReadAll(io.LimitReader(r, maxPayload+1))
	if err != nil {
		return Snapshot{}, fmt.Errorf("cache: read payload: %w", err)
	}
	if int64(len(payloadBytes)) > maxPayload {
		return Snapshot{}, fmt.Errorf("cache: payload size exceeds limit")
	}

	actualCRC := crc32.ChecksumIEEE(payloadBytes)
	if actualCRC != expectedCRC {
		return Snapshot{}, fmt.Errorf("cache: CRC mismatch (file=%08x computed=%08x)", expectedCRC, actualCRC)
	}

	entries := make([]EntrySnapshot, 0, count)
	buf := bytes.NewReader(payloadBytes)

	for i := 0; i < count; i++ {
		e, err := decodeEntry(buf, nw)
		if err != nil {
			return Snapshot{}, fmt.Errorf("cache: entry %d: %w", i, err)
		}
		entries = append(entries, e)
	}

	if buf.Len() != 0 {
		return Snapshot{}, fmt.Errorf("cache: %d trailing bytes after %d entries", buf.Len(), count)
	}

	return Snapshot{
		Version:  int(version),
		Dims:     fileDims,
		Capacity: capacity,
		Entries:  entries,
	}, nil
}


func decodeEntry(r *bytes.Reader, numWords int) (EntrySnapshot, error) {
	var keyLen uint32
	if err := binary.Read(r, binary.LittleEndian, &keyLen); err != nil {
		return EntrySnapshot{}, err
	}
	if keyLen > maxKeyLen {
		return EntrySnapshot{}, fmt.Errorf("key length %d exceeds maximum %d", keyLen, maxKeyLen)
	}
	keyBuf := make([]byte, keyLen)
	if _, err := io.ReadFull(r, keyBuf); err != nil {
		return EntrySnapshot{}, err
	}

	vecData := make([]uint64, numWords)
	for i := range vecData {
		if err := binary.Read(r, binary.LittleEndian, &vecData[i]); err != nil {
			return EntrySnapshot{}, err
		}
	}

	var ts, deadline int64
	if err := binary.Read(r, binary.LittleEndian, &ts); err != nil {
		return EntrySnapshot{}, err
	}
	if err := binary.Read(r, binary.LittleEndian, &deadline); err != nil {
		return EntrySnapshot{}, err
	}

	var valLen uint32
	if err := binary.Read(r, binary.LittleEndian, &valLen); err != nil {
		return EntrySnapshot{}, err
	}
	if valLen > maxValLen {
		return EntrySnapshot{}, fmt.Errorf("value length %d exceeds maximum %d", valLen, maxValLen)
	}
	valBuf := make([]byte, valLen)
	if _, err := io.ReadFull(r, valBuf); err != nil {
		return EntrySnapshot{}, err
	}

	var value any
	if err := json.Unmarshal(valBuf, &value); err != nil {
		return EntrySnapshot{}, err
	}

	var dl time.Time
	if deadline != 0 {
		dl = time.Unix(0, deadline)
	}

	return EntrySnapshot{
		Key:      string(keyBuf),
		VecData:  vecData,
		Value:    value,
		Ts:       time.Unix(0, ts),
		Deadline: dl,
	}, nil
}

func encodeEntry(w *bytes.Buffer, e EntrySnapshot, dims int) error {
	if len(e.VecData) != hdc.NumWords(dims) {
		return fmt.Errorf("entry %q: VecData length %d != expected %d", e.Key, len(e.VecData), hdc.NumWords(dims))
	}
	// KeyLen + Key
	keyBytes := []byte(e.Key)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(keyBytes))); err != nil {
		return err
	}
	w.Write(keyBytes)

	// VecData — raw uint64s as little-endian bytes
	for _, word := range e.VecData {
		if err := binary.Write(w, binary.LittleEndian, word); err != nil {
			return err
		}
	}

	// Timestamps
	ts := e.Ts.UnixNano()
	var deadline int64
	if !e.Deadline.IsZero() {
		deadline = e.Deadline.UnixNano()
	}
	if err := binary.Write(w, binary.LittleEndian, ts); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, deadline); err != nil {
		return err
	}

	// Value as JSON
	valJSON, err := json.Marshal(e.Value)
	if err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(valJSON))); err != nil {
		return err
	}
	w.Write(valJSON)

	return nil
}
