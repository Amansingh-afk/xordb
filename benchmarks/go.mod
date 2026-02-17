module xordb/benchmarks

go 1.22

require (
	xordb v0.0.0
	xordb/embed v0.0.0
)

require github.com/yalue/onnxruntime_go v1.13.0 // indirect

replace (
	xordb => ../
	xordb/embed => ../embed
)
