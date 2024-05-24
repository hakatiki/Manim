from manim import *


class CreateCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency
        self.play(Create(circle))  # show the circle on screen
        
class SquareToCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.rotate(PI / 4)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(FadeOut(square))  # fade out animation
        
class SquareAndCircle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set the color and transparency

        square = Square()  # create a square
        square.set_fill(BLUE, opacity=0.5)  # set the color and transparency

        square.next_to(circle, RIGHT, buff=0.5)  # set the position
        self.play(Create(circle), Create(square))  # show the shapes on screen
        
# from manim import *

class SineWave(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-PI, PI, PI / 4],
            y_range=[-1.5, 1.5, 0.5],
            axis_config={"color": BLUE},
        )
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        
        # Define the sine wave function
        sine_wave = axes.plot(lambda x: np.sin(x), color=YELLOW)
        sine_label = MathTex(r"f(x) = \sin(x)").next_to(sine_wave, UP)

        self.play(Create(axes), Write(labels))
        self.play(Create(sine_wave), Write(sine_label))
        self.wait(2)

# To run this script, save it as a .py file and execute it with Manim using:
# manim -pql your_script.py SineWave



class MobiusStrip(ThreeDScene):
    def construct(self):
        # Create axes
        axes = ThreeDAxes()

        # Define Möbius strip surface
        mobius_strip = Surface(
            lambda u, v: np.array([
                (1 + v/2 * np.cos(u/2)) * np.cos(u),
                (1 + v/2 * np.cos(u/2)) * np.sin(u),
                v/2 * np.sin(u/2)
            ]),
            u_range=[0, TAU],
            v_range=[-1, 1],
            checkerboard_colors=[RED_D, RED_E],
            resolution=(30, 64)
        )

        # Add labels
        label = Tex(r"\\text{Möbius Strip}").to_corner(UL)

        # Set camera orientation
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        # Add objects to scene
        self.add(axes, mobius_strip, label)
        
        # Create Möbius strip
        self.play(Create(mobius_strip, ))
        
        # Start camera rotation
        self.begin_ambient_camera_rotation(rate=0.3)
        
        # Wait for a while to enjoy the rotation
        self.wait(2)
        
        # Stop camera rotation
        self.stop_ambient_camera_rotation()

# To run this script, save it as a .py file and execute it with Manim using:
# manim -pql your_script.py MobiusStrip
import numpy as np

class BlackScholesSDE(Scene):
    def construct(self):
        # Parameters
        S0 = 100  # Initial stock price
        mu = 0.10  # Drift
        sigma = 0.8  # Volatility
        T = 1  # Time to maturity
        dt = 0.01  # Time step
        N = int(T / dt)  # Number of time steps
        paths = 10  # Number of paths to simulate

        # Function to simulate stock prices
        def simulate_paths(S0, mu, sigma, T, dt, N, paths):
            t = np.linspace(0, T, N)
            dB = np.sqrt(dt) * np.random.randn(paths, N)
            S = np.zeros((paths, N))
            S[:, 0] = S0
            for i in range(1, N):
                S[:, i] = S[:, i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dB[:, i-1])
            return t, S

        # Simulate stock prices
        t, S = simulate_paths(S0, mu, sigma, T, dt, N, paths)

        # Plotting stock prices
        axes = Axes(
            x_range=[0, T, T/10],
            y_range=[0, 3*S0, S0/10],
            axis_config={"color": BLUE},
            x_axis_config={"numbers_to_include": np.arange(0, T+0.1, 0.1)},
            y_axis_config={"numbers_to_include": np.arange(0, 3*S0+10, S0)},
        )
        axes_labels = axes.get_axis_labels(x_label="Time (t)", y_label="Stock Price (S)")

        self.add(axes, axes_labels)

        for path in S:
            stock_price_graph = axes.plot_line_graph(t, path, line_color=YELLOW)
            self.play(Create(stock_price_graph), run_time=2, rate_func=linear)

        self.wait(2)

        # Function for European Call and Put options
        def european_call_put(S, K, T, r, sigma):
            from scipy.stats import norm
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return call, put

        K = 100  # Strike price
        r = 0.05  # Risk-free rate

        call_prices = []
        put_prices = []
        for s in S:
            call_price, put_price = european_call_put(s[-1], K, T, r, sigma)
            call_prices.append(call_price)
            put_prices.append(put_price)

        # Display option prices
        call_price_text = Text(f"European Call Option Prices: {np.mean(call_prices):.2f}", color=GREEN).to_edge(UP)
        put_price_text = Text(f"European Put Option Prices: {np.mean(put_prices):.2f}", color=RED).next_to(call_price_text, DOWN)

        self.play(Write(call_price_text))
        self.play(Write(put_price_text))
        self.wait(2)

from manim import *
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

class EfficientFrontier(Scene):
    def construct(self):
        # Step 1: Download data
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        start_date = '2023-01-01'
        end_date = '2024-01-01'
        data = self.load_yfinance_data(tickers, start_date, end_date)
        returns = self.calculate_returns(data)
        
        # Step 2: Simulate portfolios
        results = self.efficient_frontier(returns, num_portfolios=10)
        
        # Step 3: Calculate exact efficient frontier
        desired_returns, exact_risks = self.exact_efficient_frontier(returns)
        
        # Step 4: Plot results using Manim
        self.plot_efficient_frontier(results, desired_returns, exact_risks)

    def load_yfinance_data(self, tickers, start_date, end_date):
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        return data

    def calculate_returns(self, data):
        return data.pct_change().dropna()

    def portfolio_statistics(self, weights, returns, covariance):
        portfolio_return = np.dot(weights, returns.mean()) * 252  # Annualized return
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights))) * np.sqrt(252)  # Annualized volatility
        return portfolio_return, portfolio_volatility

    def efficient_frontier(self, returns, num_portfolios=10000, risk_free_rate=0.0):
        cov_matrix = returns.cov() * 252  # Annualize the covariance matrix
        expected_returns = returns.mean() * 252  # Annualize the returns
        num_assets = len(expected_returns)
        results = np.zeros((4, num_portfolios))

        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)        
            portfolio_return, portfolio_volatility = self.portfolio_statistics(weights, returns, cov_matrix)
            results[0, i] = portfolio_volatility
            results[1, i] = portfolio_return
            results[2, i] = (portfolio_return - risk_free_rate) / portfolio_volatility  # Sharpe ratio

        return results

    def exact_efficient_frontier(self, returns):
        mu = returns.mean() * 252  # Annualize the daily returns
        Sigma = returns.cov() * 252  # Annualize the daily covariances
        Sigma_inv = np.linalg.inv(Sigma)
        ones = np.ones(len(mu))
        U = np.vstack([mu, ones]).T
        M = U.T @ (Sigma_inv @ U)
        M_inv = np.linalg.inv(M)
        desired_returns = np.linspace(mu.min(), mu.max(), 100)
        exact_risks = []

        for desired_return in desired_returns:
            u = np.array([desired_return, 1])
            w_star = Sigma_inv @ U @ M_inv @ u
            portfolio_risk = np.sqrt(np.dot(w_star.T, np.dot(Sigma, w_star))) * np.sqrt(252)
            exact_risks.append(portfolio_risk)

        return desired_returns, exact_risks

    def plot_efficient_frontier(self, results, exact_returns, exact_risks):
        axes = Axes(
            x_range=[0, max(results[0]) * 1.1, max(results[0]) / 10],
            y_range=[0, max(results[1]) * 1.1, max(results[1]) / 10],
            axis_config={"color": BLUE},
            x_axis_config={"numbers_to_include": np.arange(0, max(results[0]) * 1.1, max(results[0]) / 10)},
            y_axis_config={"numbers_to_include": np.arange(0, max(results[1]) * 1.1, max(results[1]) / 10)},
        )

        labels = axes.get_axis_labels(x_label="Volatility (Standard Deviation)", y_label="Expected Return")

        # Plot simulated portfolios
        scatter = VGroup()
        for i in range(len(results[0])):
            dot = Dot(point=axes.c2p(results[0][i], results[1][i]), color=interpolate_color(BLUE, GREEN, results[2][i] / max(results[2])), radius=0.1)
            scatter.add(dot)
            self.add(dot)
            self.wait(0.01)  # Increase speed of simulation by decreasing the wait time
        self.add(scatter)
        self.add(axes, labels, scatter)
        self.play(Create(scatter), run_time=2)

        # # Plot exact efficient frontier
        # exact_points = VGroup()
        # for x, y in zip(exact_risks, exact_returns):
        #     point = Dot(point=axes.c2p(x, y), color=RED, radius=0.05)
        #     exact_points.add(point)

        # exact_frontier = VMobject()
        # exact_frontier.set_points_as_corners([axes.c2p(x, y) for x, y in zip(exact_risks, exact_returns)])
        # exact_frontier.set_color(RED)

        # self.add(axes, labels, scatter, exact_frontier)
        # self.play(Create(exact_frontier), run_time=2)
        # self.wait(2)