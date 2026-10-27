from typing import Tuple

from imgui_bundle import imgui, implot
import numpy as np

from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState
from .plot import Plot


class HistogramPlot(Plot):
    """Stacked population by colour over time, shaded with each colour bin's creature colour."""
    def __init__(self, world: GWorld, state: GameState, default_open: bool = True):
        super().__init__('Population by Color', default_open)
        self.histogram_tracker = state.histogram_tracker
        self.n_bins = state.histogram_tracker.kwargs['n_bins']
        cmap = world.creatures.compute_hist_cmap(self.n_bins)
        # add alpha channel
        cmap = np.hstack((cmap, np.full((cmap.shape[0], 1), 255.))).astype(np.float32)
        implot.add_colormap('Histogram Colors', cmap/255.)

    def render(self, size: Tuple[float, float]):
        tracker = self.histogram_tracker
        if tracker.length == 0:
            imgui.text_disabled('(no data yet)')
            return
        updated = tracker.has_updates()
        if implot.begin_plot("##Population by Color", size,
                             flags=implot.Flags_.no_legend | implot.Flags_.no_title):  # type: ignore
            implot.setup_axes('Time', 'Population')
            implot.setup_axis_limits(implot.ImAxis_.y1, tracker.min_val,  # type: ignore
                                     tracker.max_val, cond=imgui.Cond_.always)  # type: ignore
            latest = tracker.times[-1]

            implot.setup_axis_limits(implot.ImAxis_.x1, max(0, latest-20_000), latest,  # type: ignore
                                     cond=imgui.Cond_.always if updated else imgui.Cond_.once)  # type: ignore

            # only hand implot the points inside the visible x-range (plus one on each side), since
            # plotting all n_bins series over the full history every frame gets slower as the game runs
            limits = implot.get_plot_limits()
            times = tracker.times
            lo = max(0, int(np.searchsorted(times, limits.x.min)) - 1)
            hi = int(np.searchsorted(times, limits.x.max, side='right')) + 1
            times = times[lo:hi]
            all_values = tracker.values[:, lo:hi]

            implot.push_colormap('Histogram Colors')
            for b in range(self.n_bins):
                if b == 0:
                    implot.plot_shaded(f"Color {b}", times, all_values[b])
                else:
                    implot.plot_shaded(f"Color {b}", times, all_values[b], all_values[b-1])
            implot.pop_colormap()
            implot.end_plot()
