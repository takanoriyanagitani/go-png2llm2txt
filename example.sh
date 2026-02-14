#!/bin/sh

ipng=./sample.d/input.png

geninput() {
	echo generating input image...

	mkdir -p ./sample.d

	echo 'draw a dog' |
		~/txt2llm2png \
			-width 128 \
			-height 128 \
			-model x/flux2-klein:4b-fp4 \
			-seed 101325 \
			-steps 4 |
		dd if=/dev/stdin of="${ipng}" bs=1048576 status=none

	echo
}

test -f "${ipng}" || geninput

model=ministral-3:8b-instruct-2512-q4_K_M
model=granite3.2-vision:2b-q8_0

echo reading the image...
cat "${ipng}" |
	./cmd/png2llm2txt/png2llm2txt \
		-model "${model}" \
		-prompt "describe this image in one sentence" \
		-seed 299792458 \
		-limit 16777216 \
		-temp 0.0
