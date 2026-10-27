from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from imgui_bundle import imgui, implot

from evolution.state.metric_tracker import MetricTracker


class Plot(ABC):
    """One plot shown inside the PlotsWindow. Subclasses draw themselves into a given size, so the
    window can stretch them to fit as it is resized."""
    def __init__(self, title: str, default_open: bool = True):
        self.title = title
        self.default_open = default_open

    @abstractmethod
    def render(self, size: Tuple[float, float]):
        """Draw the plot with implot at `size` (width, height) in pixels."""


class Series:
    """A line on a LinePlot: one tracker (or one component of an array-valued tracker)."""
    def __init__(self, label: str, tracker: MetricTracker, component: Optional[int] = None,
                 right_axis: bool = False):
        self.label = label
        self.tracker = tracker
        self.component = component
        self.right_axis = right_axis

    def values(self) -> np.ndarray:
        vals = self.tracker.values
        return vals if self.component is None else vals[self.component]


class LinePlot(Plot):
    """Line plot of one or more tracked metrics over time. Series marked right_axis use a second
    y axis on the right, for quantities on a different scale. Axes fit the data automatically and
    the time axis follows the latest values."""
    def __init__(self, title: str, series: List[Series], y_label: str, y2_label: str = '',
                 default_open: bool = True, hlines: Optional[List[Tuple[str, float]]] = None):
        super().__init__(title, default_open)
        self.series = series
        self.y_label = y_label
        self.y2_label = y2_label
        self.hlines = hlines or []   # (label, y) horizontal reference lines on the left axis

    def render(self, size: Tuple[float, float]):
        tracker = self.series[0].tracker
        if tracker.length == 0:
            imgui.text_disabled('(no data yet)')
            return
        flags = implot.Flags_.no_title
        if len(self.series) + len(self.hlines) == 1:
            flags |= implot.Flags_.no_legend
        if implot.begin_plot(f'##{self.title}', size, flags=flags):
            fit = implot.AxisFlags_.auto_fit
            implot.setup_axis(implot.ImAxis_.x1, 'Time', fit)
            implot.setup_axis(implot.ImAxis_.y1, self.y_label, fit)
            if any(s.right_axis for s in self.series):
                implot.setup_axis(implot.ImAxis_.y2, self.y2_label, fit | implot.AxisFlags_.opposite)
            implot.setup_legend(implot.Location_.north_west)
            times = tracker.times
            for s in self.series:
                implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y2 if s.right_axis else implot.ImAxis_.y1)
                implot.plot_line(s.label, s.tracker.times, np.ascontiguousarray(s.values()))
            implot.set_axes(implot.ImAxis_.x1, implot.ImAxis_.y1)
            for label, y in self.hlines:
                implot.plot_line(label, np.array([times[0], times[-1]]), np.array([y, y], dtype=float))
            implot.end_plot()
