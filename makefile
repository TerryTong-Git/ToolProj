clean:
	rm -rf src/exps_performance/*.txt src/exps_performance/*.png src/exps_performance/*.out

test:
	pixi run pytest --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

probs:
	pixi run pytest tests/unit/test_probs.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

utils:
	pixi run pytest tests/unit/test_utils.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

bench:
	pixi run hyperfine pytest --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

ds:
	pixi run pytest tests/unit/test_dataset.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

log:
	pixi run pytest tests/unit/test_log_results.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

metric:
	pixi run pytest tests/unit/test_metrics.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg

main:
	pixi run pytest tests/unit/test_main.py --pdb -p no:pastebin -p no:nose -p no:doctest --profile-svg
