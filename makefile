clean:
	rm -rf src/exps_performance/*.txt src/exps_performance/*.png src/exps_performance/*.out

test:
	pixi run pytest --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

arm2:
	pixi run pytest tests/unit/test_runner.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

np:
	pixi run pytest tests/unit/test_nphardeval.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

utils:
	pixi run pytest tests/unit/test_utils.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

bench:
	pixi run hyperfine pytest --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

dataset:
	pixi run pytest tests/unit/test_dataset.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

