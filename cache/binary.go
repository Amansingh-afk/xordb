package cache

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"io"
)

const (
	headerSize    = 32
	formatMagic   = "XRDB"
	formatVersion = 2
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

func encodeEntry(w *bytes.Buffer, e EntrySnapshot, dims int) error {
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
