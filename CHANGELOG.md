# Changelog

## 2026-10-16 to 2026-10-21

The `large` preset now fits on an 8 GB GPU, runs about 2× faster on an RTX 4090, and the project has a unit test suite. Three of the speed-ups change simulation behaviour slightly, so they are config options that default to the old behaviour and are switched on only in the `large` preset. Saved games keep the settings they were made with.

### Changes

| Date | Commit | What changed | Why |
| --- | --- | --- | --- |
| 2026-10-21 | `b9b9fda` | New `food_grow_every` option grows food every N steps with an N× step; 4 for `large` | Food growth passed over all 25M cells every step, the second-largest GPU cost |
| 2026-10-21 | `6dca0bc` | pytest suite of 32 tests, `pytest.ini`, README Tests section | The code had no tests; they guard the memory, kernel and performance changes |
| 2026-10-21 | `4e1fbc9` | Fix crash when births fill the buffer exactly while the creature block does not start at index 0 | Found by the new tests; long games that hit `max_creatures` could crash |
| 2026-10-21 | `78a6543` | New `brain_dtype` option stores brain weights as float16; on for `large` | Brain evaluation was about 60% of GPU time and limited by memory bandwidth |
| 2026-10-21 | `d1ff057` | New `kill_reproduce_every` option runs killing and reproduction every N steps; 4 for `large` | Kill and reproduce took about 55% of each step, mostly fixed cost per call |
| 2026-10-21 | `00a2b3c` | Reuse food-grid scratch buffers, `cumsum` instead of `conv1d`, colour plot draws only the visible range | Remove waste found by profiling, and stop the frame rate dropping in long games |
| 2026-10-21 | `3790815` | `--style cuda-events` reads `creatures.traits.algos` | The benchmark crashed after an earlier refactor moved `algos` |
| 2026-10-17 | `95ff72c` | Ignore `*.sqlite` | `nsys stats` writes a large SQLite file into the repo folder |
| 2026-10-16 | `bc1d502` | Ignore six unused sprite images | They are early drafts that no code loads |
| 2026-10-16 | `5911556` | Ignore `venv/` and `*.qdstrm`; commit three sprite images | Fresh clones crashed without the images |
| 2026-10-16 | `af2e9d7` | README section estimating effort for the planned features | Planning the feature list |
| 2026-10-16 | `33e9a85` | `large` `max_creatures` 262,144 → 131,072; new `--max_creatures` flag | The `large` preset needed about 9 GiB of GPU memory |

### GPU memory

Every creature trait is allocated up front for `max_creatures`, not for the creatures alive. Each brain (130 inputs → 50 → 40 → 12) is about 9,000 weights, so 262,144 creatures needed about 9 GiB. At 131,072 the measured peak is 4.9 GiB (2.8 GiB at 65,536). `--max_creatures` overrides the cap for a new game; a loaded checkpoint keeps its saved config.

On a machine with a separate display GPU, the simulation can run on another card while the window stays on the display card, as long as both are visible to CUDA (the game shares buffers between CUDA and OpenGL). For example, where CUDA device 0 is the compute card: `CUDA_VISIBLE_DEVICES=0,1 ./launch.py`.

### Performance

Measured on the `large` preset, RTX 4090, no window. Runs vary by a few percent, because the eat kernel's atomic adds make runs non-deterministic even with a fixed seed.

| Change | Measurement | Result | Behaviour change |
| --- | --- | --- | --- |
| Quick fixes (`00a2b3c`) | ~170 → 172–184 steps/s | ~2–5% | None: identical results to the original code |
| Kill and reproduce every 4 steps (`d1ff057`) | 179 → 237 steps/s | ~32% | Dead creatures linger, and ready creatures wait, up to 3 extra steps |
| Float16 brain weights (`78a6543`) | 287 → 389 steps/s from a shared 70k-creature snapshot | ~35% | Brain outputs differ by 4×10⁻⁴ on average, up to 0.09 |
| Food growth every 4 steps (`b9b9fda`) | 351–356 → 388 steps/s from a shared 86k-creature snapshot | ~10% | Food grows in jumps every 4 steps; mean food level unchanged |

- **Quick fixes:** `eat_grow` keeps its two full-size scratch grids between steps and re-zeroes only the cells creatures touched. Memory compaction counts living creatures in a sliding window with `cumsum` instead of a 1.9 ms cuDNN `conv1d`. The population-by-colour plot passes implot only the points in the visible time range.
- **`kill_reproduce_every`:** measured 215, 237 and 246 steps/s at 2, 4 and 8.
- **`brain_dtype`:** brain evaluation reads every creature's weights each step at the card's memory-bandwidth limit, so float16 halves its cost and the weights' memory (4.38 → 2.19 GiB at 131,072 creatures). Mutations are computed in float32 and rounded on write; biases and activations stay float32.
- **`food_grow_every`:** on steps between growth passes, only the eating updates are applied, at creatures' cells. The cell info panel shows zero growth on those steps.

After these changes the GPU is busy most of each step, so removing the remaining CPU overhead (for example by packing all traits into one array) would gain at most about 35%, and less with more creatures.

### Crash fix (`4e1fbc9`)

When births brought the population to exactly `max_creatures`, `add_with_deaths_max_array` collected only the free slots inside and to the right of the creature block, then moved the block start to 0. If the block did not start at 0, which is normal after deaths shift it, the left-hand slots were missed, so there were fewer slots than children and writing them failed with a shape mismatch. The selected creature's index was also not shifted. Both are fixed; `test_filling_buffer_when_block_does_not_start_at_zero` reproduces the original crash.

### Tests

Run with `python3 -m pytest` (35 tests, about 8 s on an RTX 4090; GPU tests are skipped without CUDA).

| File | What it checks |
| --- | --- |
| `tests/test_creature_array.py` | Every creature's traits stay together through births and deaths for all three packing strategies, a random run of updates, the selected creature, and the crash case |
| `tests/test_kernels.py` | Eat and food-growth kernels against a plain-PyTorch reference (normal, crowded, batched growth); scratch buffers re-zeroed; colour mapping |
| `tests/test_simulation.py` | 200-step invariants for each option combination; scheduling; float16 brains; checkpoint round trips |
| `tests/test_cpu_utils.py` | Publish/subscribe, metric trackers, quantization, config helpers |

### Known issues

- `kernels/eat.cu` computes a per-creature food-cover cost (`creature_food_decr`) but removes a flat `food_cover_decr` per creature instead, unlike the config comments and the cell info panel. Fixing it would change game balance, so it is left as is; the kernel tests match the current behaviour.
- `GWorld.from_checkpoint` calls `torch.load` with `weights_only=False`, so loading a checkpoint can run arbitrary code. Only load checkpoints you trust.
