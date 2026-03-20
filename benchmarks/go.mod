module github.com/Amansingh-afk/xordb/benchmarks

go 1.22

require (
	github.com/Amansingh-afk/xordb v0.0.0
	github.com/Amansingh-afk/xordb/embed v0.0.0
)

require github.com/yalue/onnxruntime_go v1.13.0 // indirect

replace (
	github.com/Amansingh-afk/xordb => ../
	github.com/Amansingh-afk/xordb/embed => ../embed
)
