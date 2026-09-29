.PHONY: doctor data train sft bench serve test

doctor:
	python -m terraforge doctor

data:
	python -m terraforge data --config configs/default.yaml

train:
	python -m terraforge train --config configs/default.yaml

sft:
	python -m terraforge sft --config configs/default.yaml

bench:
	python -m terraforge bench --config configs/default.yaml

serve:
	python -m terraforge serve --config configs/default.yaml

test:
	python -m unittest discover -s tests -t .
