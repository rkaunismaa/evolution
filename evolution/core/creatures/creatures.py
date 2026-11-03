from typing import Tuple, Union, List
import torch
from torch import Tensor
import numpy as np
# import logging
# logging.basicConfig(level=logging.ERROR, filename='game.log')

from evolution.core.benchmarking import Profile
from evolution.cuda.cu_algorithms import CUDAKernelManager
from evolution.core.config import Config
from evolution.state.game_state import GameState

from .creature_array import CreatureArray
from .creature_trait import CreatureTrait
from . import trait_initializer as init
from .trait_initializer import Initializer
from . import trait_normalizer as norm
from .trait_normalizer import Normalizer


class Creatures:
    """
    This class includes all CreatureTrait definitions, and the logic for determining
    which creatures are dead, which are reproducing, and how to update the state of the
    world based on the actions of the creatures. It passes references to all its CreatureTraits
    to a CreatureArray object, which manages the memory of each trait.
    """

    def __init__(self, cfg: Config, kernels: CUDAKernelManager, device: torch.device):
        self.cfg = cfg
        self.kernels = kernels

        # how much indexing into the food grid needs to be adjusted to avoid OOB
        self.pad = self.cfg.food_sight

        self.posn_bounds = (0., cfg.size-1e-3)
        self.num_food = (cfg.food_sight*2+1)**2

        self.device = device
        self.offsets = torch.tensor([[i, j] for i in range(-self.pad, self.pad+1)
                                     for j in range(-self.pad, self.pad+1)], device=self.device).unsqueeze(0)
        self.activations = []
        self._eat_scratch: Tuple[Tensor, Tensor] | None = None  # (pct_eaten, food_grid_updates), see eat_grow
        self.n_births = 0   # running totals since the world was created/loaded, for the GUI plots
        self.n_deaths = 0
        self.traits = CreatureArray(cfg, device)
        self.generate_from_cfg()

    def generate_from_cfg(self):
        """Definition of all creature Traits for root creature. This should only be called by the
        root Creatures object. When reproduction happens, it should create a new CreatureArray object,
        which is mostly just a container, not a Creatures object, which has specifics about how
        what particular traits exist and what they do. Each time a new CreatureTrait attribute is set,
        it is automatically added to self.traits.variables, which allows for iterating over all traits that
        exist and applying operations to them. For lists of CreatureTraits, each element of the list
        is also added to self.traits.variables."""
        self.sizes = CreatureTrait(tuple(),
                        Initializer.mutable('uniform_', *self.cfg.init_size_range),
                        Normalizer(norm.clamp, *self.cfg.size_range),
                        self.device)

        color_channels = 1
        self.colors = CreatureTrait(tuple(),
                          Initializer.mutable('uniform_', 1, 255),
                          Normalizer(norm.clamp, 1, 255), self.device)

        self.visual_colors = CreatureTrait((3,),
                                 Initializer.other_dependent('colors', init.cuda_hsv_spiral, self.kernels),
                                 None, self.device)

        # the initializer here isn't fully correct, as it should be normal(0, 1) for the first 2 dimensions and
        # uniform on the last dimension, but I guess that's what you get for combining 2 traits that really should have been separate
        self.rays = CreatureTrait((self.cfg.num_rays, 3),
                        Initializer.mutable('normal_', 0, self.cfg.ray_dist_range[1]),
                        Normalizer(norm.normalize_rays, self.cfg),
                        self.device)


        self.positions = CreatureTrait((2,),
                             Initializer.force_mutable('uniform_', self.cfg.reproduce_dist, *self.posn_bounds),
                             Normalizer(norm.clamp, *self.posn_bounds),
                             self.device)

        self.energies = CreatureTrait(tuple(),
                            Initializer.other_dependent('sizes', self.cfg.init_energy),
                            None, self.device)

        self.reproduce_energies = CreatureTrait(tuple(),
                                      Initializer.other_dependent('sizes', self.cfg.reproduce_thresh),
                                      None, self.device)

        self.healths = CreatureTrait(tuple(),
                           Initializer.other_dependent('sizes', self.cfg.init_health),
                           None, self.device)

        self.eat_pcts = CreatureTrait(tuple(),
                            Initializer.other_dependent('sizes', init.eat_amt, self.cfg),
                            None, self.device)

        self.age_speeds = CreatureTrait(tuple(),
                              Initializer.other_dependent('sizes', self.cfg.age_speed_size),
                              None, self.device)

        self.age_mults = CreatureTrait(tuple(),
                             Initializer.fillable('fill_', 1.0),
                             None, self.device)

        self.memories = CreatureTrait((self.cfg.mem_size,),
                            Initializer.fillable('zero_'),
                            None, self.device)

        self.ages = CreatureTrait(tuple(),
                        Initializer.fillable('zero_'),
                        None, self.device)

        self.n_children = CreatureTrait(tuple(),
                              Initializer.fillable('zero_'),
                              None, self.device)

        self.head_dirs = CreatureTrait((2,),
                             Initializer.fillable('normal_', 0, 1),
                             Normalizer(norm.normalize_head_dirs), self.device)

        # input: rays (num_rays*color_dims) + memory (mem_size) + num_food ((2*food_sight+1)**2) + health (1) + energy (1) +
        #         head_dir (2) + age_mult (1) + color (color_dims) + size (1)
        # output: move (1) + rotate (1) + memory (mem_size)
        layer_sizes = [
            self.cfg.num_rays*color_channels + self.cfg.mem_size + self.num_food + 1 + 1 + 2 + 1 + color_channels + 1,
            *self.cfg.brain_size,
            self.cfg.mem_size + 2
        ]
        weights = []
        biases = []
        brain_dtype = getattr(torch, self.cfg.brain_dtype)
        for prev_layer, next_layer in zip(layer_sizes[:-1], layer_sizes[1:]):
            w_norm = 0.5 * np.sqrt(15. / prev_layer)
            b_norm = 0.5 / np.sqrt(next_layer)

            weights.append(CreatureTrait((prev_layer, next_layer),
                               Initializer.mutable('uniform_', -w_norm, w_norm),
                               None, self.device, dtype=brain_dtype))

            biases.append(CreatureTrait((1, next_layer),
                              Initializer.mutable('uniform_', -b_norm, b_norm),
                              None, self.device))

        self.weights: List[CreatureTrait] = weights
        self.biases: List[CreatureTrait] = biases

        # this must be defined last, so that the right number of mutable parameters are set
        self.mutation_rates = CreatureTrait((self.traits.num_mutable+1,),
                                  Initializer.mutable('uniform_', *self.cfg.init_mut_rate_range),
                                  None, self.device)


        self.traits.generate_from_cfg()

    @property
    def population(self) -> int:
        return self.traits.population

    def __setattr__(self, name, value):
        """Capture any CreatureTrait objects that are defined, and register them with self.traits,
        so that CreatureArray can manage their memory. Additionally, we also can track how many
        mutable parametrs have been added so far, so that each one can be assigned a unique
        mut_idx when reproducing."""
        if isinstance(value, CreatureTrait):
            self.traits.register_trait(name, value)
        elif isinstance(value, list) and all(isinstance(v, CreatureTrait) for v in value):
            for i, v in enumerate(value):
                self.traits.register_trait(name + f"_{i}", v)
        super().__setattr__(name, value)

    def fused_kill_reproduce(self, central_food_grid: Tensor, state: GameState):
        """Kill the dead creatures and reproduce the living ones."""
        dead, num_dead, dead_idxs = self._kill_dead(central_food_grid)  # determine who is dead
        alive = ~dead  # type: ignore
        new_creatures = self._reproduce(alive, num_dead)  # determine who is reproducing
        self.n_deaths += num_dead
        self.n_births += new_creatures.population if new_creatures is not None else 0
        if new_creatures or num_dead > 0:   # rearrange memory and update current data
            self.traits.add_with_deaths(dead_idxs, alive, num_dead, new_creatures, state)

    @Profile.cuda_profile
    def food_types(self, idxs: Tensor | None = None) -> Tensor:
        """int32 [N] (or [len(idxs)]) index of the food grid each creature eats from: colours [1, 255]
        split into cfg.food_types equal bands, so similarly coloured creatures share a food."""
        colors = self.colors if idxs is None else self.colors[idxs]
        n_types = self.cfg.food_types
        if n_types == 1:
            return torch.zeros(colors.shape[0], dtype=torch.int32, device=self.device)
        return ((colors - 1.0) * (n_types / 254.0)).int().clamp_(0, n_types - 1)

    def _kill_dead(self, central_food_grid: Tensor) -> Tuple[Tensor, int, Tensor]:
        """If any creature drops below 0 health or energy, kill it. Update the food grid by depositing an amount of food
        that depends on the creature's size at death.

        Returns:
            dead: [N] boolean tensor indicating which creatures are dead
            num_dead: number of creatures that died
            dead_idxs: [num_dead] indices of the dead creatures
        """
        if self.cfg.immortal:
            return (torch.zeros(self.population, dtype=torch.bool, device=self.device),
                    # torch.tensor(0, device=self.device, dtype=torch.int32),
                    0,
                    torch.tensor([], device=self.device, dtype=torch.int32))

        health_deaths = self.healths <= 0
        energy_deaths = self.energies <= 0
        dead = health_deaths | energy_deaths   # contiguous boolean tensor

        dead_idxs = dead.nonzero().squeeze(1)
        num_dead = dead_idxs.shape[0]

        if num_dead > 0:
            dead_posns = self.positions[dead_idxs].long()
            # food_grid[dead_posns[..., 1], dead_posns[..., 0]] += self.cfg.dead_drop_food(self.sizes[self.dead])
            # dead creatures leave food of their own type
            central_food_grid.index_put_((self.food_types(dead_idxs).long(), dead_posns[..., 1], dead_posns[..., 0]),
                                self.cfg.dead_drop_food(self.sizes[dead_idxs]),
                                accumulate=True)
        else:
            dead_idxs = torch.tensor([], device=self.device, dtype=torch.int32)

        return dead, num_dead, dead_idxs

    @Profile.cuda_profile
    def _get_reproducers(self, alive: Union[None, Tensor]) -> Tensor:
        """Return an index tensor (into current memory) indicating which creatures are able to reproduce.
        Creatures must be in the 'mature' aging stage (between adolescent and old), and have enough
        energy to reproduce."""
        # current mem boolean tensor
        reproducers = alive & (self.ages >= self.age_speeds*self.cfg.age_mature_mul) & (self.energies >= self.reproduce_energies)
        return reproducers.nonzero().squeeze(1)

    @Profile.cuda_profile
    def _reduce_reproducers(self, reproducers: Tensor, num_reproducers: int,
                            max_reproducers: int) -> Tuple[Tensor, int]:
        if num_reproducers > max_reproducers:  # this benefits older creatures because they
            num_reproducers = max_reproducers  # are more likely to be in the num_reproducers window
            perm = torch.randperm(reproducers.shape[0], device=self.device)
            reproducers = reproducers[perm[:num_reproducers]]
        return reproducers, num_reproducers

    @Profile.cuda_profile
    def _reproduce(self, alive: Tensor, num_dead: int) -> Union[None, 'CreatureArray']:
        """For sufficiently high energy creatures, create a new creature with mutated genes. Returns the set
        of new creatures in a CreatureArray object."""
        max_reproducers = self.cfg.max_creatures + num_dead - self.population # dont bother if none can reproduce anyway
        if max_reproducers <= 0:
            return

        reproducers = self._get_reproducers(alive)  # current memory boolean tensor
        # limit reproducers so we don't exceed max size
        num_reproducers = reproducers.shape[0]
        if num_reproducers == 0:  # no one can reproduce
            return

        partners = None
        if getattr(self.cfg, 'sexual_reproduction', 0):
            mothers, fathers = self._find_mates(reproducers)
            num_pairs = mothers.shape[0]
            if 2 * num_pairs > max_reproducers:   # keep a random subset of the pairs that fits
                keep = torch.randperm(num_pairs, device=self.device)[:max_reproducers // 2]
                mothers, fathers = mothers[keep], fathers[keep]
                num_pairs = mothers.shape[0]
            # creatures with no mate breed alone with probability mate_fallback_prob, if there is room
            solo = reproducers[:0]
            fallback = getattr(self.cfg, 'mate_fallback_prob', 0.0)
            room = max_reproducers - 2 * num_pairs
            if fallback > 0 and room > 0:
                unpaired = torch.ones(self.population, dtype=torch.bool, device=self.device)
                unpaired[mothers] = False
                unpaired[fathers] = False
                solo = reproducers[unpaired[reproducers]]
                solo = solo[torch.rand(solo.shape[0], device=self.device) < fallback][:room]
            if num_pairs == 0 and solo.shape[0] == 0:
                return
            # two children per pair: each parent pays for (and is the 'reproducer' of) one of them.
            # A creature breeding alone is its own partner, so crossover leaves it a plain copy
            reproducers = torch.cat([mothers, fathers, solo])
            partners = torch.cat([fathers, mothers, solo])
            num_reproducers = reproducers.shape[0]
        else:
            reproducers, num_reproducers = self._reduce_reproducers(reproducers, num_reproducers, max_reproducers)
        self.n_children[reproducers] += 1

        # could fuse this
        self.energies[reproducers] -= self.sizes[reproducers]  # subtract off the energy that you've put into the world
        self.energies[reproducers] /= self.cfg.reproduce_energy_loss_frac  # then lose a bit extra because this process is lossy

        new_creatures = self.traits.reproduce_traits(self.mutation_rates[reproducers], reproducers, num_reproducers,
                                                     partners)
        return new_creatures

    def _find_mates(self, candidates: Tensor) -> Tuple[Tensor, Tensor]:
        """Pair up creatures that are ready to reproduce (`candidates`, an index tensor). Two can
        mate if they are in the same mate_range-wide square of the world and their colours differ by
        at most mate_color_range. Candidates are sorted by square and then colour, and each pairs with
        the next one in that order, so similar colours pair first. Returns two index tensors of equal
        length (each pair once); creatures left over have no mate this time."""
        empty = candidates[:0]
        if candidates.shape[0] < 2:
            return empty, empty
        pos = self.positions[candidates]
        cells_per_side = int(self.cfg.size // self.cfg.mate_range) + 1
        cell = (pos[:, 1] / self.cfg.mate_range).long() * cells_per_side + (pos[:, 0] / self.cfg.mate_range).long()
        colors = self.colors[candidates]
        order = torch.argsort(cell.double() * 256.0 + colors.double())   # colours are in [1, 255]
        cell, colors, cands = cell[order], colors[order], candidates[order]

        # rank within each square: position minus the index where that square's run starts
        n = cands.shape[0]
        idx = torch.arange(n, device=self.device)
        new_run = torch.ones(n, dtype=torch.bool, device=self.device)
        new_run[1:] = cell[1:] != cell[:-1]
        run_start = torch.cummax(torch.where(new_run, idx, torch.zeros_like(idx)), dim=0).values
        rank = idx - run_start

        first = idx[:-1][(rank[:-1] % 2 == 0)]     # each even-ranked creature pairs with the next one,
        second = first + 1                         # if it is in the same square and close in colour
        ok = (cell[first] == cell[second]) & ((colors[second] - colors[first]).abs() <= self.cfg.mate_color_range)
        return cands[first[ok]], cands[second[ok]]

    def eat_grow(self, food_grid: Tensor, state: GameState, capacity_mult: Tensor | None = None):
        """Compute how much food each creature eats, by taking a fixed percentage of the available
        food in the cell it is in. If there are more creatures in a cell than the reciprocal of
        this fixed percentage (i.e. there isn't enough food to go around), then each creature
        instead gets an equal share of the total food in the cell.

        After eating, we apply food_cover costs to the cell. This represents something like the
        presence of creatures making the environment less hospitable for food growth / more toxic.
        This is a fixed cost per creature in the cell.

        Finally, we allow food in all cells to grow by an amount proportional to its distance from
        the 'maximum' food level. Food levels beyond the maximum food level are decayed
        proportionally to how high above the maximum they are."""
        state.selected_creature.extract_pre_eat_state()
        state.selected_cell.extract_pre_grow_state()

        pos = self.positions.int() + self.pad
        food_types = self.food_types()
        n_cells = food_grid.shape[-1]   # food_grid is [food_types, S, S], S including padding

        # pct_eaten and food_grid_updates are only ever written at creatures' cells, so we keep them
        # around between steps and re-zero just those cells at the end, rather than allocating and
        # zero-filling two full-size grids every step
        if self._eat_scratch is None or self._eat_scratch[0].shape != food_grid.shape:
            self._eat_scratch = (torch.zeros_like(food_grid, dtype=torch.float32),
                                 torch.zeros_like(food_grid, dtype=torch.float32))
        pct_eaten, food_grid_updates = self._eat_scratch

        # calculate how many creatures are in each position, useful for computng how much food each creature eats
        threads_per_block = 512
        blocks_per_grid = self.population // threads_per_block + 1
        self.kernels('setup_eat_grid', blocks_per_grid, threads_per_block,
                     pos, food_types, self.eat_pcts, pct_eaten, self.population, n_cells)

        # calculate how much food each creature eats, add 1 to ages, update energies, take away living costs
        alive_costs = torch.zeros(self.population, device=self.device, dtype=torch.float32)
        self.kernels('eat', blocks_per_grid, threads_per_block,
                     pos, food_types, self.eat_pcts, pct_eaten, self.sizes, self.age_speeds, food_grid,
                     food_grid_updates, alive_costs, self.energies, self.healths, self.ages, self.age_mults,
                     self.population, n_cells, self.cfg.food_cover_decr)

        # grow food, apply eating costs
        # step_size = (torch.sum(alive_costs)/self.cfg.max_food/(self.cfg.size**2)).item()
        # step_size = (self.cfg.max_food / (self.cfg.size**(1.9)))#.item()
        touched = (food_types.long() * n_cells + pos[:, 1].long()) * n_cells + pos[:, 0].long()
        if state.time % self.cfg.food_grow_every == 0:
            # growing touches every cell, so we only do it every food_grow_every steps, with a
            # proportionally larger step (the grow kernel also applies this step's eating updates)
            step_size = self.cfg.food_step_size * self.cfg.food_grow_every
            threads_per_block = (16, 16)
            blocks_per_grid = (n_cells // threads_per_block[0] + 1,
                               n_cells // threads_per_block[1] + 1,
                               food_grid.shape[0])   # one z-block per food type

            self.kernels('grow', blocks_per_grid, threads_per_block,
                         food_grid_updates, food_grid, n_cells, self.pad, step_size, self.cfg.max_food_per_type(),
                         capacity_mult if capacity_mult is not None else 0)   # 0 = null pointer
        else:
            # no growth this step: just apply the eating updates, which are only at creatures' cells
            # (duplicate indices all write the same value, so cells with several creatures are fine)
            step_size = 0.0
            flat = food_grid.view(-1)
            flat[touched] = flat[touched] - food_grid_updates.view(-1)[touched]

        state.selected_cell.extract_post_grow_state(pct_eaten, food_grid_updates, step_size, pos, food_types)
        state.selected_creature.extract_post_eat_state(pos, alive_costs)

        pct_eaten.view(-1).index_fill_(0, touched, 0.)
        food_grid_updates.view(-1).index_fill_(0, touched, 0.)


    def extract_food_windows(self, indices, food_grid, food_types):
        """Indices is [N, 2] indices into food_grid.
        Food grid is [T, S, S], where T is the number of food types and S the padded width of the world.
        food_types is [N], the food grid each creature sees.
        Returns a [N, W] tensor of the food in the W x W window around each index,
        which goes as an input into creatures' brains."""
        windows = indices.unsqueeze(1) + self.offsets  # [N, 1, 2] + [1, 9, 2] = [N, 9, 2]
        # if windows.min() < 0 or windows.max() >= food_grid.shape[0]:
        #     raise ValueError("OOB")
        if food_grid.shape[0] == 1:
            return food_grid[0, windows[..., 1], windows[..., 0]]
        return food_grid[food_types.long().unsqueeze(1), windows[..., 1], windows[..., 0]]


    def collect_stimuli(self, collisions, food_grid):
        """Collisions: [N, R], food_grid: [T, W, W] (T food types, W = padded world size) -> [N, F],
        where F is the number of features that get passed in to the creatures' brains."""
        # get the food in the creature's vicinity
        # we must add self.pad, because positions are stored in central_food_grid coordinates,
        # but we need to index into food_grid with food_grid coordinates (which includes padding)
        posn = self.positions.long() + self.pad
        food = self.extract_food_windows(posn, food_grid, self.food_types())  # [N, (2*sight+1)**2]
        rays_results = collisions

        return torch.cat([rays_results,  # type: ignore
                          self.memories,
                          food,
                          self.healths.unsqueeze(1),
                          self.energies.unsqueeze(1),
                          self.head_dirs,
                          self.age_mults.unsqueeze(1),
                          self.colors.unsqueeze(1),
                          self.sizes.unsqueeze(1),
                        ],
                        dim=1)

    def forward(self, inputs):
        """Inputs: [N, 1, F] tensor.
        Compute the forward pass of the neural network brain for each creature, and update
        their memories by the output of the network.
        Returns: `outputs` [N, 2+cfg.mem_size], where the first coordinate is how much
            to move forward/backward the 2nd coordinate is how much to rotate rotate left/rotate,
            and the last cfg.mem_size coordinates are for memory."""

        self.activations.clear()
        for w, b in zip(self.weights, self.biases):
            self.activations.append(inputs)
            inputs = torch.tanh(inputs.to(w.dtype) @ w + b)  # b is float32, so this promotes back to float32
        outputs = inputs.squeeze(dim=1)  # [N, O]
        self.activations.append(inputs)
        self.memories[:] = outputs[:, 2:]  # only 2 outputs actually do something, so the rest are memory
        return outputs

    def rotate_creatures(self, outputs: Tensor, state: GameState):
        """Given the neural network outputs `outputs`, rotate each creature accordingly.
        Rotation is a scalar between -1 and 1, which is combined with the creatures' size
        to compute the number of radians that are rotated. Negative rotation is leftwards,
        positive is rightwards. We also take a small energy penalty for rotating."""
        # rotation => need to rotate the rays and the object's direction

        state.selected_creature.extract_pre_rotate_state()

        block_size = 512
        grid_size = self.population // block_size + 1
        rotation_matrix = torch.empty((self.population, 2, 2), device=self.device)  # [N, 2, 2]
        self.kernels('build_rotation_matrices', grid_size, block_size,
                     outputs, self.sizes, self.energies, self.population, outputs.shape[1],
                     rotation_matrix)

        # rotate the rays and head directions
        if self.cfg.rays_rotate:
            self.rays[..., :2] = self.rays[..., :2] @ rotation_matrix  # [N, 32, 2] @ [N, 2, 2]
        self.head_dirs[:] = (self.head_dirs.unsqueeze(1) @ rotation_matrix).squeeze(1)  # [N, 1, 2] @ [N, 2, 2]

        state.selected_creature.extract_post_rotate_state(outputs[:,1], rotation_matrix)

    def move_creatures(self, outputs: Tensor, state: GameState):
        """Given the neural network outputs `outputs`, move each creature accordingly.
        Moving forward is a scalar between -1 and 1, which is combined with the creatures' size
        to compute the distance that is moved. Positive scalars are movement forward, and negative
        scalars are backwards movement. We also take a small energy penalty for moving.
        We clamp the position to stay inside the grid, but penalize the creature for the full
        movement anyway."""
        # move => need to move the object's position
        # maybe add some acceleration/velocity thing instead
        move = self.cfg.move_amt(outputs[:,0], self.sizes)
        move_cost = self.cfg.move_cost(move, self.sizes)

        self.energies -= move_cost
        self.positions += self.head_dirs * move.unsqueeze(1)   # move the object's to new position
        self.positions.normalize()# = torch.clamp(self.positions, *self.posn_bounds)  # don't let it go off the edge

        state.selected_creature.extract_move_state(outputs[:,0], move, move_cost)


    def do_attacks(self, attacks: Tensor, state: GameState):
        """Attacks is [N, 2], where the 1st coordinate is an integer indicating how many things the
        organism is attacking, and the 2nd coordinate is a float indicating the amount of damage the
        organism is taking from other things attacking it. We update each
        creature's health and energy accordingly."""

        attack_cost = self.cfg.attack_cost(attacks[:,0], self.sizes)
        self.energies -= attack_cost
        self.healths -= attacks[:,1]

        state.selected_creature.extract_attack_state(attacks, attack_cost)

    def total_energy(self):
        """Return the sum of energies of all creatures"""
        return torch.sum(self.energies).item() # type: ignore

    def compute_hist_cmap(self, n_bins: int) -> np.ndarray:
        """Return a histogram of the sizes of the creatures"""
        midpoints = torch.linspace(1, 255, n_bins, device=self.device)
        cmap = init.cuda_hsv_spiral(midpoints.unsqueeze(1), self.kernels)
        return cmap.cpu().numpy()

    def histogram(self, n_bins: int) -> np.ndarray:
        """Return a histogram of the sizes of the creatures"""
        return torch.cumsum(torch.histc(self.colors, bins=n_bins, min=1, max=255), dim=0).cpu().numpy()  # type: ignore

