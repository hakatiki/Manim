from manim import *

class MarkowitzTheory(Scene):
    def construct(self):
        title = Text("Markowitz Portfolio Theory").scale(1.2)
        self.play(Write(title))
        self.wait(2)
        self.play(title.animate.to_edge(UP))

        intro_text = Tex(
            "Markowitz Portfolio Theory, also known as Modern Portfolio Theory (MPT), \n"
            "is a framework for constructing a portfolio of assets that maximizes \n"
            "expected return for a given level of risk."
        ).shift(UP).scale(0.7)
        self.play(Write(intro_text))
        self.wait(3)
        self.play(FadeOut(intro_text))

        # Define a portfolio
        portfolio_text = Tex(
            "A portfolio is a combination of assets, each with a weight $w_i \\in \\mathbb{R}$. \n"
            "The weights represent the proportion of the total investment in each asset, \n"
            "and they sum to 1."
        ).shift(UP).scale(0.7)
        self.play(Write(portfolio_text))
        self.wait(3)

        # Portfolio weights equation
        weights_eq1 = MathTex(
            "w_1 + w_2 + \\cdots + w_n = 1"
        ).next_to(portfolio_text, DOWN).scale(0.9)
        weights_eq2 = MathTex(
            "\\sum_{i=1}^{n} w_i = 1"
        ).next_to(portfolio_text, DOWN).scale(0.9)
        weights_label = Tex("Sum of Portfolio Weights (weights can be negative)").next_to(weights_eq2, DOWN).scale(0.6)

        self.play(Write(weights_eq1))
        self.wait(2)
        self.play(Transform(weights_eq1, weights_eq2))
        self.play(FadeIn(weights_label))
        self.wait(3)
        self.play(FadeOut(weights_eq1, weights_label, portfolio_text))

        # The expected return equation with a description
        return_intro = Tex(
            "The portfolio return $R_p$ is the weighted sum of the returns of the individual assets."
        ).shift(UP).scale(0.7)
        self.play(Write(return_intro))
        self.wait(3)
        
        return_eq1 = MathTex(
            "R_p = w_1 R_1 + w_2 R_2 + \\cdots + w_n R_n"
        ).scale(0.9).next_to(return_intro, DOWN)
        return_eq2 = MathTex(
            "R_p = \\sum_{i=1}^{n} w_i R_i"
        ).scale(0.9).next_to(return_intro, DOWN)
        return_label = Tex("Portfolio Return").next_to(return_eq2, DOWN).scale(0.6)

        self.play(Write(return_eq1))
        self.wait(2)
        self.play(Transform(return_eq1, return_eq2))
        self.play(FadeIn(return_label))
        self.wait(3)
        self.play(FadeOut(return_eq1, return_label, return_intro))

        # The risk equation with explanations for sigma and rho
        risk_intro = Tex(
            "The risk or variance of the portfolio $\\sigma_p^2$ is given by:"
        ).shift(UP).scale(0.7)
        self.play(Write(risk_intro))
        self.wait(3)
        
        risk_eq = MathTex(
            "\\sigma_p^2 = \\sum_{i=1}^{n} \\sum_{j=1}^{n} w_i w_j \\sigma_i \\sigma_j \\rho_{ij}"
        ).scale(0.9).next_to(risk_intro, DOWN)
        risk_label = Tex("Portfolio Variance").next_to(risk_eq, DOWN).scale(0.6)
        sigma_explanation = Tex("$\\sigma_i$: Standard deviation of asset $i$").next_to(risk_eq, DOWN).scale(0.6).shift(DOWN).set_color(BLUE)
        rho_explanation = Tex("$\\rho_{ij}$: Correlation between assets $i$ and $j$").next_to(sigma_explanation, DOWN).scale(0.6).set_color(GREEN)

        self.play(Write(risk_eq))
        self.play(FadeIn(risk_label))
        self.wait(2)
        self.play(FadeIn(sigma_explanation))
        self.wait(2)
        self.play(FadeIn(rho_explanation))
        self.wait(3)
        self.play(FadeOut(risk_eq, risk_label, sigma_explanation, rho_explanation, risk_intro))

        # Efficient frontier explanation
        efficient_frontier_text = Tex(
            "The efficient frontier is a set of optimal portfolios that offer \n"
            "the highest expected return for a defined level of risk."
        ).scale(0.7)
        self.play(FadeOut(title))
        efficient_frontier_text.to_edge(UP)
        self.play(Write(efficient_frontier_text))
        self.wait(3)

# To run this script, save it as a .py file and execute it with Manim using:
# manim -pql your_script.py MarkowitzTheory
