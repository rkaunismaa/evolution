# Changelog

## 2026-11-03

The `large` preset now gives creatures a 5×5 food view instead of 9×9 (`food_sight` 4 → 2), because the navigation experiments showed they were not using it. At 131,072 creatures this lowers peak GPU memory from 3.09 to 2.27 GiB and step time from 5.05 to 3.82 ms on an RTX 4090. Saved games keep the `food_sight` they were made with, so the saving applies only to new games.

The cell info panel now shows the right food-cover cost in games saved before `food_cover_scaled` existed.

Species-specific food is built and off by default: `food_types` (or `--food_types` for a new game) splits colours into that many bands, and each band eats, sees and leaves behind only its own food. The types share the world's food (`food_types_share=1`, the default), so species compete for it: in a 40,000-step test all three survived, food limited the population throughout, and the lead changed hands.

A mass extinction now ends the game with the "Mass extinction!" message instead of crashing.

Sexual reproduction is built and off by default (`sexual_reproduction`, or `--sexual` for a new game): nearby, similarly coloured creatures pair up, and their children mix the parents' traits, taking whole brain neurons from one parent or the other.

| Date | Commit | What changed | Why |
| --- | --- | --- | --- |
| 2026-11-03 | `415d8cf` | Sexual reproduction: children inherit mutation rates whole from the paying parent | Mixed mutation rates rose ~9× and nearly tripled the death rate |
| 2026-11-03 | `6156b8b` | New `sexual_reproduction` option and `--sexual` flag: mates by area and colour, two children per pair, whole-neuron crossover | README planned feature: sexual reproduction |
| 2026-11-03 | `709ef93` | New `food_types_share` option, on by default: each type grows towards `max_food / food_types` | Without it, extra types added food and the species barely competed (see the comparison below) |
| 2026-11-03 | `f4f69b7` | Fix crash when every creature dies in one step; `step()` reports the extinction | The block-moving code sliced `csum[:-0]`, an empty tensor; the original `conv1d` failed too |
| 2026-11-03 | `990e858` | Plots window kept inside the game window | A layout saved on a larger screen opened it partly off-screen |
| 2026-11-03 | `33c6f6b` | New `food_types` option and `--food_types` flag: several food grids, one per colour band | README planned feature: species-specific food, so different niches can co-exist |
| 2026-11-03 | `d402813` | Cell panel shows the flat food-cover cost in older games; new test checks the panel against the eat kernel | The panel always showed the scaled cost, so its cover, eaten and growth amounts were wrong for those games |
| 2026-11-03 | `9e91182` | `food_sight=2` in the `large` preset; first brain layer 130 → 74 inputs | Blinding food vision changed nothing over 60,000 steps, and the view was 62% of the first layer's weights |

## 2026-10-16 to 2026-10-27

The `large` preset now fits on an 8 GB GPU, runs about 2× faster on an RTX 4090, the project has a unit test suite, new games use the documented food-cover cost, and the `large` preset has rich and poor regions of land. Three of the speed-ups change simulation behaviour slightly, so they are config options that default to the old behaviour and are switched on only in the `large` preset. Saved games keep the settings they were made with.

### Changes

| Date | Commit | What changed | Why |
| --- | --- | --- | --- |
| 2026-10-27 | `5f3230f` | `--run` flag starts the game running instead of paused | The game always starts paused (space toggles it) |
| 2026-10-27 | `773dcc6` | One movable, resizable Plots window; new population, mean food, mean size and age, and births/deaths plots | README planned features: a plot manager and more plots |
| 2026-10-22 | `b891914` | Capacity field scaled by its spread instead of its maximum | Restores the contrast the `large` preset was tuned with (std 0.19, ~15% rich land) |
| 2026-10-22 | `d2a5dcc` | Capacity map rebuilt as a wave field that can drift (`food_capacity_change_steps`, off by default) | README planned feature: large-scale, slow environmental change |
| 2026-10-22 | `c0fa69a` | `food_capacity_variation=0.5` in the `large` preset | First planned feature from the README, turned on after a comparison |
| 2026-10-22 | `baa0cc1` | New `food_capacity_variation` / `food_capacity_scale` options give each cell its own maximum food from a smooth random map | README planned feature: uneven food to reward navigation |
| 2026-10-22 | `d2aa419` | `food_cover_scaled=1` in the `small`, `medium` and `large` presets | New games get the intended cost and a correct cell panel; old saves keep the flat cost |
| 2026-10-22 | `b1f818f` | New `food_cover_scaled` option applies the per-creature food-cover cost | `eat.cu` computed the documented cost but removed a flat cost instead |
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

### Food cover cost (`b1f818f`, `d2aa419`)

`kernels/eat.cu` computed a per-creature food-cover cost, `food_cover_decr × food_cover_decr_pct × eat_pct`, as documented in `config.py` and shown in the cell info panel, but removed a flat `food_cover_decr` per creature instead. The scaled variable was added in `2762aa9` without switching the line that removes the food over to it.

`food_cover_scaled=1` applies the scaled cost. The presets set it, but the `Config` default stays 0, so older checkpoints keep the flat cost they were made with.

| 20,000-step `large` runs, 2 seeds each | Flat cost | Scaled cost |
| --- | --- | --- |
| Mean food level (max 15) | ~10.5, slowly falling | ~13.3, steady |
| Mean creature size | ~0.79, slowly rising | ~0.70, steady |
| Population reaches the 131,072 cap | ~step 5,000 | by step 2,500 |

Small creatures, which eat a small share of their cell, destroy up to 100× less food with the scaled cost, so being small is cheaper. In both modes the population then stays at the cap.

### Uneven food capacity (`baa0cc1`, `c0fa69a`)

With `food_capacity_variation` above 0, each cell's `max_food` is multiplied by a smooth random map spanning about [1 − variation, 1 + variation], with regions roughly `food_capacity_scale` cells wide (default 250). The map multiplies `cfg.max_food`, so the Max Food slider still works, and it is saved in checkpoints. At 0, the default, no map is allocated and the grow kernel behaves exactly as before. The `large` preset uses 0.5; the map costs about 100 MB of GPU memory and no measurable speed.

| 20,000-step `large` runs, 2 seeds each, variation 0.5 | Normal brains | Movement shuffled between creatures (control) |
| --- | --- | --- |
| Creatures on rich land (capacity > 1.2, 15–16% of the area) | 29–33% | 37–39% |
| Mean capacity under creatures (map mean 1.0) | 1.10–1.11 | 1.14–1.15 |
| Mean creature size | 0.70–0.73 | 0.40–0.41 |

Mean food (13.2) and creature size match uniform-capacity runs. Creatures gather on rich land, but the control shows this comes from births and survival there rather than steering: creatures whose movement is decoupled from what they see gather even more. Within 20,000 steps there is no evidence yet that creatures evolve to navigate towards rich land; with the population pinned at the `max_creatures` cap, selection may be too weak.

### Drifting land (`d2a5dcc`, `b891914`)

The capacity map is now a `FoodCapacityField`: a sum of 24 plane waves with integer wavevectors, so it tiles across the world's edges and averages exactly 1. It is scaled so its standard deviation is 0.375 × `food_capacity_variation`, keeping ~99% of land within 1 ± variation. With `food_capacity_change_steps` above 0, each wave drifts at its own speed, so rich and poor regions grow, shrink, merge and split, becoming unrecognisable after about that many steps. The field is recomputed in place every `food_capacity_update_every` steps (default 50; 1.4 ms at size 5000), and checkpoints store only the wave parameters. Drift is off by default and in all presets.

| 20,000-step `large` runs, 2 seeds each, variation 0.5 | Static, normal | Static, control | Drifting (10,000 steps), normal | Drifting, control |
| --- | --- | --- | --- | --- |
| Creatures on rich land (14–15% of area) | 25–26% | 30–32% | 13% | 12–13% |
| Mean capacity under creatures (map mean 1.0) | 1.08–1.09 | 1.11–1.12 | 0.99 | 0.98 |
| Mean food level (max 15) | 13.2 | 14.1 | 14.5 | 15.9 |

With drifting land, populations stay put as the land moves and end up on formerly rich land, while newly rich areas go ungrazed. Normal brains do no better than the control, so creatures do not steer. In every run food is near its maximum and the population is pinned at `max_creatures`, so there is little pressure to seek food; making food rather than the cap limit the population is the likely next step.

### Plots window (`773dcc6`)

The two fixed-position plot windows are replaced by one movable, resizable **Plots** window. Plots stack under collapsible headers, stretch to the window's width and share its height, and the **Show** menu picks which appear. ImGui remembers the window's position and size in `imgui.ini`; without a saved layout it starts at the bottom right.

| Plot | Shows | Open at start |
| --- | --- | --- |
| Population | Creature count, with the `max_creatures` cap marked | Yes |
| Energy over time | log10 of total energy (unchanged) | Yes |
| Population by Color | Stacked population by colour (unchanged) | No |
| Mean food level | Average food per cell, with `max_food` marked | No |
| Mean size and age | Average size, and average age on a second axis | No |
| Births and deaths | Births and deaths per step | No |

Each new plot has a tracker saved in checkpoints. Births and deaths are counted on the CPU from values the simulation already has, and the new trackers cost no measurable speed. Which plots are shown is not saved between sessions.

### Species-specific food (`33c6f6b`)

`food_types=T` stores the food as one `[T, S, S]` grid. A creature's colour (1–255) picks its type: the colours are split into `T` equal bands, so similarly coloured creatures, which already do little damage to each other, share a food.

- **Eating, vision and death** use only the creature's own food, so the brain inputs do not change.
- **Each type grows independently**, with its own rich and poor land when `food_capacity_variation > 0`. The wave fields are uncorrelated between types.
- **GUI:** the cell panel lists the food of each type and totals the per-step amounts. The creature panel shows which food the creature eats. The heatmap shows the selected creature's food, or the mean over all types.
- **Checkpoints** save the grid as `[T·S, S]` so that quantization keeps one scale per column. Older saves load as one food type.
- **Cost on the `large` preset:** with one type, behaviour matches the previous code to float rounding and step time is unchanged (3.85 ms). Three types take 4.59 ms per step and 0.9 GiB more memory.

Test runs on the `large` preset with three foods:

| Setting | Population | Creatures per type (start → 10,000 → 40,000 steps) | Mean food per type at the end (max 15) |
| --- | --- | --- | --- |
| Default growth, 131,072 cap, 30,000 steps | At the cap from step 2,500 | 2,755 / 2,701 / 2,736 → 45,143 / 45,278 / 40,651 → 47,557 / 45,006 / 38,509 (at 30,000) | 14.3 / 14.3 / 14.4 |
| Growth 1.0, 262,144 cap, 40,000 steps | Fell to 3,734 by step 2,500, at the cap by step 10,000 | 2,755 / 2,701 / 2,736 → 103,321 / 91,812 / 67,011 → 93,766 / 83,012 / 85,366 | 6.9 / 7.3 / 7.3 |

- **The species co-exist and balance each other.** With scarce food, the largest species held the least food per cell and lost ground, so the smallest species grew from 26% to 33% of the population.
- **Shared food (`709ef93`):** at first each type grew at the full rate, so `T` types put `T` times as much food in the world. Now, by default (`food_types_share=1`), `max_food` and `init_food_scale` are totals, and each type grows towards `max_food / T`. Set `food_types_share=0` for the full `max_food` per type. The comparison below led to this default.

Same settings (growth 1.0, cap 262,144, 40,000 steps, `food_sight=2`):

| | One food, `max_food` 15 | Three foods, `max_food` 15 | Three foods, `max_food` 5 (same total food) |
| --- | --- | --- | --- |
| Lowest population | 2,902 | 3,734 | 1,323 |
| When food limits the population | Steps ~15,000–25,000; at the cap otherwise after step 10,000 | Until steps 7,500–10,000, then at the cap | All 40,000 steps (138,269 at the end) |
| Creatures per type at the end | 262,144 | 93,766 / 83,012 / 85,366 | 33,522 / 45,815 / 58,932 |
| Mean food at the end | 3.5 of 15 | 6.9–7.3 of 15 | 3.5–3.7 of 5 |

With shared food the species compete for the whole run and the lead changes hands: type 2 held up to 51% of creatures (step 22,500), and type 3 overtook it after step 35,000.
- Default-growth runs stay limited by the cap, so the species barely compete there.

### Sexual reproduction (`6156b8b`)

With `sexual_reproduction=1`, creatures that are ready to reproduce (mature, with enough energy) look for a mate:

1. **Finding mates:** candidates are sorted by which `mate_range`-wide square of the world they are in (default 50), then by colour. Each one pairs with the next in that order if both are in the same square and their colours differ by at most `mate_color_range` (default 20). Every creature mates at most once per round, and the code is plain PyTorch with no new kernel.
2. **Children:** each pair has two. Each parent pays for one, as in asexual reproduction, and that child starts near it. If there is room for fewer children, a random subset of pairs breeds.
3. **Crossover:** a child's brain takes each neuron, meaning its incoming weights and its bias, whole from one parent; averaging weights tends to produce broken brains. Other inherited traits mix per element (whole rays, single values for size and colour), and then mutate. Mutation rates are the exception: they come whole from the paying parent (`415d8cf`, see below).
4. **No mate:** the creature keeps its energy and tries again next round.

The asexual path is unchanged, and older saves keep reproducing asexually.

Test runs on the `large` preset, 30,000 steps, seed 0. The mate-range columns were measured before the mutation-rate fix; the last column is the current code:

| | Asexual | Mate range 20 | Mate range 50 | Mate range 100 | Mate range 50, current code |
| --- | --- | --- | --- | --- | --- |
| Population at step 5,000 | 131,072 (cap) | 21,667 | 54,983 | 131,072 (cap) | 104,488 |
| At the cap from | by step 5,000 | step 10,000–15,000 | step 5,000–10,000 | by step 5,000 | step 5,000–10,000 |
| Births by step 30,000 | 1.53 M | 4.14 M | 4.66 M | 3.75 M | 2.08 M |
| Mean creature size at the end | 0.72 | 0.71 | 0.84 | 0.88 | 0.63 |

- **Early on, finding a mate is the bottleneck.** While the population is small, ready creatures are spread thin. With range 20, about 1,600 creatures were ready per round at step 10,000, but only about 240 pairs formed.

**Why creatures died young, and the fix (`415d8cf`).** In the first version, births at the cap were 2.5–3 times the asexual rate, so creatures died that much more often. To find out why, each part of reproduction was switched off in turn. These are deaths over steps 15,000–20,000 on the `large` preset:

| Children mix... | Deaths per step | Mean age at death | Killed by attacks | Died before maturity | Median mutation rate |
| --- | --- | --- | --- | --- | --- |
| Nothing: asexual | 47.6 | 2,777 | 6.8% | 3.5% | 0.038 |
| Nothing, but sexual pairing and two children per pair | 51.1 | 2,539 | 4.4% | 3.3% | |
| Brain neurons only | 56.1 | 2,310 | 5.9% | 5.0% | |
| Sizes only | 50.7 | 2,565 | 4.2% | 3.3% | |
| Colours only | 50.7 | 2,562 | 4.3% | 3.3% | |
| Rays only | 51.7 | 2,507 | 4.9% | 3.5% | |
| Mutation rates only | 98.3 | 1,331 | 41.6% | 19.1% | |
| Everything (first version) | 136.7 | 945 | 53.3% | 25.3% | 0.350 |
| Everything except mutation rates (current) | 55.8 | 2,323 | 5.4% | 4.5% | 0.030 |

- **Mixing mutation rates let them run away.** In an asexual lineage, a low mutation rate stays tied to the good genes it protects, so selection holds rates down. Crossover breaks that link, and the median rate rose about 9 times. Heavily mutated children could not steer, ran into others and were killed in attacks.
- **Brain crossover by neuron is nearly harmless**, as is mixing the other traits.
- Children now take their mutation rates whole from the paying parent.

### Navigation experiments (no code changes)

Measurements on the `large` preset with `food_capacity_variation=0.5`, to see whether creatures evolve to seek food. "Control" shuffles move and rotate outputs between creatures each step, so movement cannot depend on what a creature senses; "food-blind" shuffles each creature's 9×9 food view with another creature's, so food vision carries no information.

- **Gathering on rich land is not steering.** With static land, creatures are about 1.7× over-represented on rich land, but the control gathers even more (about 2.1×), so the effect comes from births and survival there. With drifting land the effect disappears for both.
- **Food is abundant in the presets.** In every default-growth run mean food sat near 13–14 of a possible 15 and the population was pinned at `max_creatures`. Even at a 262,144 cap, growth rates of 5 and 2.5 (default 10) stayed cap-limited.
- **Scarce food (growth 1.0, cap 262,144, 60,000 steps)** limits the population for roughly the first 30,000 steps, with an early crash to about 3,000 creatures and boom-and-bust swings. Creatures then evolve smaller (mean size 0.64 → 0.49) and the population returns to the cap with mean food around 3.6.

| 60,000 steps, growth 1.0, cap 262,144, seed 0 | Normal | Control (movement shuffled) | Food-blind |
| --- | --- | --- | --- |
| Lowest population | 3,220 | 12 | 4,441 |
| Population at the end | 262,144 (cap) | 104,804 | 262,144 (cap) |
| Mean food at the end (max 15) | 3.6 | 11.7 | 3.6 |
| Mean creature size at the end | 0.49 | 0.28 | 0.48 |
| Creatures on rich land (14.6% of area) | 20% | swings 9–29% | 20% |

- **Brains matter, food vision does not.** Shuffling movement is nearly fatal, but blinding food vision changes nothing, so the useful behaviour comes from other inputs (own energy, health, size, memory, or what the rays see). Likely reasons navigation has not evolved: neighbouring cells differ little (regions are ~250 cells wide, the view is 9×9), moving costs energy, and 60,000 steps allow relatively few generations.
- **Trade-off (acted on 2026-11-03, `food_sight=2`):** with `food_sight=4` the food view is 81 of each brain's 130 inputs (about 62% of the first layer's weights). `food_sight=1` (9 inputs) would roughly halve brain memory and cut brain evaluation, the largest GPU cost, by about a third, at the price of making navigation even harder to evolve.

### Crash fix (`4e1fbc9`)

When births brought the population to exactly `max_creatures`, `add_with_deaths_max_array` collected only the free slots inside and to the right of the creature block, then moved the block start to 0. If the block did not start at 0, which is normal after deaths shift it, the left-hand slots were missed, so there were fewer slots than children and writing them failed with a shape mismatch. The selected creature's index was also not shifted. Both are fixed; `test_filling_buffer_when_block_does_not_start_at_zero` reproduces the original crash.

### Tests

Run with `python3 -m pytest` (82 tests, about 18 s on an RTX 4090; GPU tests are skipped without CUDA).

| File | What it checks |
| --- | --- |
| `tests/test_creature_array.py` | Every creature's traits stay together through births and deaths for all three packing strategies, a random run of updates, the selected creature, and the crash case |
| `tests/test_kernels.py` | Eat and food-growth kernels against a plain-PyTorch reference (normal, crowded, batched growth, flat and scaled cover cost, capacity map, one and three food types); scratch buffers re-zeroed; colour mapping |
| `tests/test_simulation.py` | 200-step invariants for each option combination; scheduling; float16 brains; checkpoint round trips; capacity field range, smoothness, convergence, drift and saving; plot trackers and birth/death rates |
| `tests/test_food_types.py` | Colour bands, grid shapes, vision, eating and dead creatures' food limited to the own type, separate rich land per type, 200-step invariants, checkpoint round trips, older capacity-field saves, cell panel totals, shared and full food per type |
| `tests/test_sexual_reproduction.py` | Mate finding (same square, close colour, at most one mate each), whole-neuron crossover masks and children, two children per pair with each parent paying, waiting without a mate, 200-step invariants, older configs staying asexual |
| `tests/test_cpu_utils.py` | Publish/subscribe, metric trackers, quantization, config helpers, older saved configs keeping original behaviour |

### Known issues

- `GWorld.from_checkpoint` calls `torch.load` with `weights_only=False`, so loading a checkpoint can run arbitrary code. Only load checkpoints you trust.

## Earlier history: June to November 2024

The original project by JacksonKaunismaa: 147 commits from 2024-06-15 to 2024-11-01, ending at `928df79`. The project had no changelog then, so this is a summary of the commit messages, grouped by period; `git log 928df79` has the full detail.

| Period | Highlights |
| --- | --- |
| June 2024 | Initial commit (`4273535`, 2024-06-15). CUDA ray tracer for creature vision and the food grid; vectorized creature operations; a much faster ray tracer that chunks the world into cells (`b7e4f11`); gridded attack checking, about 2× faster (`5a608b1`); a `Config` object for world parameters. |
| July 2024 | Kernels moved to CUDA-python for OpenGL interop (`0c505c8`); a working window with CUDA-python, PyTorch and ModernGL sharing data (`90c6155`); instanced creature rendering and a camera. |
| Early August 2024 | Click-and-drag navigation and world coordinates; ray and hitbox visualization; a camera that follows a creature; brain and "thoughts" visualization; saving the game; speed controls. |
| Mid August 2024 | Faster reproduction and killing through batched random numbers and index-based memory rearranging (`indexed-reproduce-kill` branch, merged in `1967423`); a creature size maximum; fused CUDA kernels for rotation and for eating and food growth; an eating rework against many-creature exploits; cell selection; ageing and an age requirement for reproduction. |
| Late August 2024 | `CreatureArray` and minimal-copy reproduction; eating scaled by creature size (`2762aa9`); health effects of eating, faster regrowth of negative food, attacks scaled by distance; publisher-subscriber updates for visuals; an old-age penalty and quantized saving for large games (`cc61467`); a GUI with imgui and glfw, with game, creature and cell panels; size-dependent ageing speed. |
| September 2024 | `imgui_bundle` and implot; generic `CreatureTraits` that can depend on other traits (`471f670`); hyperparameter search for algorithm parameters; more brain inputs (head direction, age, colour, size) and ray lengths scaled by size (`e2d2a0c`); total-energy plot and scalar trackers saved in checkpoints; pylint-driven refactors; scalar colours mapped to RGB in a CUDA kernel; PyTorch and nsys profiling (`fa53620`); population-by-colour histogram plot (`bba3cb3`); command-line options for `launch.py`. |
| October to November 2024 | README and docstrings; config refactor; `Creatures` no longer subclasses `CreatureArray` (`6ca1ea3`); creature status panel fix (`428b1ec`, 2024-10-09); GPLv3 license added (`928df79`, 2024-11-01). |
