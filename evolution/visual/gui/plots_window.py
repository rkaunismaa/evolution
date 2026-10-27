from typing import Dict, List

from imgui_bundle import imgui
from moderngl_window import BaseWindow

from evolution.core.config import Config
from evolution.core.gworld import GWorld
from evolution.state.game_state import GameState
from .ui_element import Window
from .plots.plot import Plot, LinePlot, Series
from .plots.histogram_plot import HistogramPlot


class PlotsWindow(Window):
    """Plot manager: one movable, resizable window holding every plot. Plots stack vertically under
    collapsible headers and stretch to the window's width; open plots share its height. The Show menu
    picks which plots appear. ImGui remembers the window's position and size in imgui.ini."""
    MIN_PLOT_HEIGHT = 120 / 13   # in multiples of the font size
    PLACING_FRAMES = 60

    def __init__(self, cfg: Config, world: GWorld, window: BaseWindow, state: GameState):
        super().__init__('Plots', self.PLOT_WIDTH)
        self.wnd = window
        s = state
        self.plots: List[Plot] = [
            LinePlot('Population', [Series('Population', s.population_tracker)], 'Creatures',
                     hlines=[('Max creatures', float(cfg.max_creatures))]),
            LinePlot('Energy over time', [Series('Energy', s.energy_tracker)], 'log10(Energy)'),
            HistogramPlot(world, state, default_open=False),
            LinePlot('Mean food level', [Series('Food', s.food_tracker)], 'Food per cell',
                     hlines=[('Max food', float(cfg.max_food))], default_open=False),
            LinePlot('Mean size and age', [Series('Size', s.size_tracker),
                                           Series('Age', s.age_tracker, right_axis=True)],
                     'Size', y2_label='Age (steps)', default_open=False),
            LinePlot('Births and deaths', [Series('Births / step', s.birth_death_tracker, component=0),
                                           Series('Deaths / step', s.birth_death_tracker, component=1)],
                     'Per step', default_open=False),
        ]
        self.shown: Dict[str, bool] = {p.title: True for p in self.plots}
        # Without a saved layout, keep placing the window for the first frames: the game window
        # starts at its configured size and is only then resized/maximised by the window manager.
        self._placing_frames = 0 if self._has_saved_layout() else self.PLACING_FRAMES

    def render(self):
        font = imgui.get_font_size()
        win_w, win_h = self.wnd.size
        if self._placing_frames > 0:   # default layout: a column at the bottom right
            self._placing_frames -= 1
            size = (font * self.PLOT_WIDTH, win_h * 0.6)
            imgui.set_next_window_pos((win_w - size[0], win_h - size[1]), cond=imgui.Cond_.always)
            imgui.set_next_window_size(size, cond=imgui.Cond_.always)

        with self.begin(imgui.WindowFlags_.menu_bar):
            if not self.open:
                return
            if imgui.begin_menu_bar():
                if imgui.begin_menu('Show'):
                    for p in self.plots:
                        clicked, value = imgui.menu_item(p.title, '', self.shown[p.title])
                        if clicked:
                            self.shown[p.title] = value
                    imgui.end_menu()
                imgui.end_menu_bar()

            shown = [p for p in self.plots if self.shown[p.title]]
            for p in shown:
                if p.default_open:
                    imgui.set_next_item_open(True, imgui.Cond_.once)
                if imgui.collapsing_header(p.title):
                    # open plots split the height left below the remaining headers
                    remaining_headers = len(shown) - shown.index(p) - 1
                    header_h = imgui.get_frame_height_with_spacing()
                    avail_w, avail_h = imgui.get_content_region_avail()
                    open_after = sum(1 for q in shown[shown.index(p) + 1:] if self._is_open(q))
                    height = (avail_h - remaining_headers * header_h) / (1 + open_after) \
                        - imgui.get_style().item_spacing.y
                    p.render((avail_w, max(height, font * self.MIN_PLOT_HEIGHT)))

    def _has_saved_layout(self) -> bool:
        """Whether imgui.ini already holds a position and size for this window."""
        path = imgui.get_io().get_ini_filename()
        try:
            with open(path, encoding='utf-8') as f:
                return f'[Window][{self.title}]' in f.read()
        except (OSError, TypeError):
            return False

    @staticmethod
    def _is_open(p: Plot) -> bool:
        """Whether p's collapsing header is open (ImGui keeps that state per window)."""
        storage = imgui.get_state_storage()
        default = 1 if p.default_open else 0
        return bool(storage.get_int(imgui.get_id(p.title), default))
