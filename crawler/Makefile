max_seq ?= 200
seq := $(shell (seq 0 $(max_seq)))
pieces := $(seq:%=../data_200/%page.csv)

all: clawer_200

../data_200/%page.csv:
	python3 crawler_sofifa.py -n $*

clawer_200: $(pieces)

clean:
	rm ../data_200/*page.csv
