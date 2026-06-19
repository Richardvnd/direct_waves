import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class PlotConfig:
    def __init__(self):
        self.style = "stylesheet.mplstyle"
        self.fig_width = 246.0 * (1.0 / 72.27)
        self.fig_width_2 = 510.0 * (1.0 / 72.27)
        self.fig_height = self.fig_width / 1.618
        self.fig_height_2 = self.fig_width_2 / 1.618
        self.colors = [
            "#5e2174",
            "#5b4a9a",
            "#4480a5",
            "#32a8a3",
            "#4ccf8e",
            "#a4e954",
            "#e6e033"
        ]
        self.colors2 = [
            "#8B5FBF",
            "#A15C9E",
            "#C26C88",
            "#D87570",
            "#DE6A5E",
        ]
        self.colormap = LinearSegmentedColormap.from_list(
            "custom_colormap", self.colors
        )
        self.colormap2 = LinearSegmentedColormap.from_list(
            "custom_colormap2", self.colors2
        )
        self.color_dw = "#b2182b"

    def apply_style(self):
        plt.style.use(self.style)
