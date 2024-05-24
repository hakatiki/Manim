from manim import *
import numpy as np

# Function to create correlated asset returns
def create_correlated_returns(timesteps,std, rho):
    np.random.seed(42)
    trend = np.exp(np.linspace(0, 2, timesteps))  # Exponential trend
    noise1 = np.random.normal(0, std, timesteps)  # Larger noise
    noise2 = np.random.normal(0, std, timesteps)
    asset1_returns = trend + noise1
    asset2_returns = trend + rho * noise1 + np.sqrt(1 - rho**2) * noise2
    portfolio_returns = 0.5 * asset1_returns + 0.5 * asset2_returns
    return asset1_returns, asset2_returns, portfolio_returns


class CorrelationEffect(Scene):
    def construct(self):
        title = Text("Understanding Correlation in Portfolios").scale(1.2)
        self.play(Write(title))
        self.wait(2)
        self.play(title.animate.to_edge(UP))

        # Plotting function
        def plot_returns(asset1_returns, asset2_returns, portfolio_returns, title_text):
            axes = Axes(
                x_range=[0, len(asset1_returns), 10],
                y_range=[0, max(np.max(asset1_returns), np.max(asset2_returns), np.max(portfolio_returns)) + 1, 1],
                x_length=6,
                y_length=3,
                axis_config={"color": BLUE}
            ).shift(UP * 0.5)

            labels = axes.get_axis_labels(x_label="Time", y_label="Returns")

            self.play(Create(axes), Write(labels))

            asset1_line = axes.plot_line_graph(range(len(asset1_returns)), asset1_returns, add_vertex_dots=False, line_color=YELLOW)
            asset2_line = axes.plot_line_graph(range(len(asset2_returns)), asset2_returns, add_vertex_dots=False, line_color=GREEN)
            portfolio_line = axes.plot_line_graph(range(len(portfolio_returns)), portfolio_returns, add_vertex_dots=False, line_color=RED)

            self.play(Create(asset1_line), Write(Text("Asset 1").next_to(asset1_line, UP).scale(0.5)))
            self.play(Create(asset2_line), Write(Text("Asset 2").next_to(asset2_line, DOWN).scale(0.5)))
            self.play(Create(portfolio_line), Write(Text("50-50 Portfolio").next_to(portfolio_line, UP).scale(0.5)))
            self.wait(2)

            scenario_title = Text(title_text).next_to(axes, DOWN)
            self.play(Write(scenario_title))
            self.wait(1)
            self.play(
                *[FadeOut(mob) for mob in self.mobjects]
                # All mobjects in the screen are saved in self.mobjects
            )
            return axes, asset1_line, asset2_line, portfolio_line, scenario_title

        # High Positive Correlation
        asset1_returns, asset2_returns, portfolio_returns = create_correlated_returns(100, 0.5, 0.9)
        high_pos_objects = plot_returns(asset1_returns, asset2_returns, portfolio_returns, "High Positive Correlation (0.9)")

        # High Negative Correlation
        asset1_returns, asset2_returns, portfolio_returns = create_correlated_returns(100, 0.5, -0.9)
        high_neg_objects = plot_returns(asset1_returns, asset2_returns, portfolio_returns, "High Negative Correlation (-0.9)")

        # Zero Correlation
        asset1_returns, asset2_returns, portfolio_returns = create_correlated_returns(100, 0.5, 0.0)
        zero_corr_objects = plot_returns(asset1_returns, asset2_returns, portfolio_returns, "Zero Correlation (0.0)")

        self.wait(3)

        explanation = Text(
            "The correlation between the returns of two assets affects the overall portfolio return.\n"
            "In these examples, we see how the portfolio returns change based on the correlation."
        ).scale(0.7).to_edge(UP)

        self.play(Write(explanation))
        self.wait(3)

# To run this script, save it as a .py file and execute it with Manim using:
# manim -pql your_script.py CorrelationEffect
